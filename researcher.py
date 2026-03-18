#!/usr/bin/env python3
"""LLM-driven experiment orchestrator following karpathy/autoresearch pattern.

Core loop:
  1. LLM proposes experiment config based on program.md + results.tsv + git log
  2. Write .env file and commit
  3. Run experiment (start server -> evaluate -> stop server)
  4. Score results and append to results.tsv
  5. If improved: keep commit. If degraded: git reset to discard.
  6. Repeat until budget exhausted or LLM declares convergence.

Git history becomes the research memory — each kept commit is a
successful iteration.
"""
import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_composite_score(result: dict) -> float:
    """Compute weighted composite score from a result file.

    Weights (from program.md):
      Speed:   30% (accuracy + gen t/s normalized)
      Coding:  30% (accuracy + response time)
      Context: 20% (needle accuracy + pp t/s normalized)
      Agentic: 20% (accuracy + concurrent throughput)
    """
    card = result.get("score_card", {})
    score = 0.0

    if "speed" in card:
        acc = card["speed"].get("accuracy", 0)
        gen_tps = min(card["speed"].get("avg_gen_tps", 0) / 100, 1.0)
        score += 0.30 * ((acc * 0.6) + (gen_tps * 0.4))

    if "coding" in card:
        acc = card["coding"].get("accuracy", 0)
        elapsed = card["coding"].get("avg_elapsed", 30)
        speed_factor = max(0, 1.0 - (elapsed / 60))
        score += 0.30 * ((acc * 0.6) + (speed_factor * 0.4))

    if "context" in card:
        needle = card["context"].get("needle_accuracy", 0)
        pp_tps = min(card["context"].get("avg_pp_tps", 0) / 3000, 1.0)
        score += 0.20 * ((needle * 0.6) + (pp_tps * 0.4))

    if "agentic" in card:
        acc = card["agentic"].get("accuracy", 0)
        conc = card["agentic"].get("concurrent_accuracy", 0)
        score += 0.20 * ((acc * 0.5) + (conc * 0.5))

    return round(score, 4)


# ---------------------------------------------------------------------------
# Results TSV
# ---------------------------------------------------------------------------

TSV_HEADER = [
    "step", "experiment", "ctx_size", "cache_k", "cache_v",
    "batch", "ubatch", "parallel",
    "coding_grade", "agentic_grade", "speed_grade", "context_grade",
    "gen_tps", "pp_tps", "composite_score", "kept", "analysis",
]


def init_results_tsv(path: Path):
    """Create results.tsv with header if it doesn't exist."""
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(TSV_HEADER)


def append_results_tsv(path: Path, row: dict):
    """Append a row to results.tsv."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row.get(h, "") for h in TSV_HEADER])


def read_results_tsv(path: Path) -> str:
    """Read results.tsv as string for LLM context."""
    if not path.exists():
        return "(no results yet)"
    return path.read_text()


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def git_run(*args, cwd=None):
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd, capture_output=True, text=True
    )
    return result.stdout.strip(), result.returncode


def git_commit(message: str, files: list, cwd=None):
    """Stage files and commit."""
    for f in files:
        git_run("add", f, cwd=cwd)
    git_run("commit", "-m", message, cwd=cwd)


def git_reset_last(cwd=None):
    """Discard the last commit (keep files unstaged)."""
    git_run("reset", "HEAD~1", cwd=cwd)
    git_run("checkout", ".", cwd=cwd)


def git_current_branch(cwd=None) -> str:
    """Get current branch name."""
    out, _ = git_run("branch", "--show-current", cwd=cwd)
    return out


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a systems performance researcher running an automated experiment loop \
to optimize llama-server configuration for local LLM inference.

RESEARCH PROGRAM:
{program}

SEARCH SPACE:
{search_space}

GPU: {gpu}
MODEL: {model}
MEMORY: {memory}

You will receive the full results.tsv history and must propose the next \
experiment. Follow the autoresearch pattern:
- Change ONE or TWO parameters per experiment
- Build on what worked, avoid what failed
- When 3+ consecutive experiments show no improvement, declare convergence

RESPONSE FORMAT — respond with ONLY valid JSON:
{{
  "analysis": "2-3 sentence analysis of results so far",
  "reasoning": "Why this specific experiment (1-2 sentences)",
  "converged": false,
  "experiment": {{
    "name": "short_descriptive_name",
    "ctx_size": 262144,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "batch_size": 2048,
    "ubatch_size": 512,
    "parallel": 1,
    "gpu_layers": 100,
    "threads": 0,
    "extra_flags": ""
  }},
  "recommendation": null
}}

When converged=true, set recommendation to your final best config (same format \
as experiment) and explain why in analysis.\
"""


def build_system_prompt(script_dir: Path, gpu: str, model: str,
                        memory: str) -> str:
    """Build the system prompt from program.md and search_space.json."""
    program = (script_dir / "program.md").read_text()
    search_space = json.loads((script_dir / "search_space.json").read_text())

    return SYSTEM_PROMPT_TEMPLATE.format(
        program=program,
        search_space=json.dumps(search_space["parameters"], indent=2),
        gpu=gpu,
        model=model,
        memory=memory,
    )


def query_llm(host: str, port: int, system_msg: str,
              user_msg: str) -> str:
    """Query the research LLM."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 2000,
        "temperature": 0.3,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()


def parse_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    # Try code blocks first
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if match:
        return json.loads(match.group(1))

    # Try raw JSON
    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(response[start:end])

    raise ValueError(f"No JSON found in response:\n{response[:500]}")


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def write_env_file(config: dict, path: Path):
    """Write experiment .env from config dict."""
    lines = [f"# Auto-generated by researcher.py"]
    lines.append(f"CTX_SIZE={config['ctx_size']}")
    lines.append(f"CACHE_TYPE_K={config['cache_type_k']}")
    lines.append(f"CACHE_TYPE_V={config['cache_type_v']}")
    lines.append(f"BATCH_SIZE={config['batch_size']}")
    lines.append(f"UBATCH_SIZE={config['ubatch_size']}")
    lines.append(f"PARALLEL={config['parallel']}")
    lines.append(f"GPU_LAYERS={config.get('gpu_layers', 100)}")
    if config.get("threads", 0) > 0:
        lines.append(f"THREADS={config['threads']}")
    lines.append(f"EXTRA_FLAGS=\"{config.get('extra_flags', '')}\"")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_experiment(script_dir: Path, env_file: Path) -> bool:
    """Run experiment via run_single.sh."""
    result = subprocess.run(
        [str(script_dir / "run_single.sh"), str(env_file)],
        cwd=str(script_dir),
    )
    return result.returncode == 0


def load_result(path: Path) -> dict:
    """Load a result JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch loop for llama-server optimization"
    )
    parser.add_argument("--research-host", default="",
                        help="LLM host for research queries")
    parser.add_argument("--research-port", type=int, default=0,
                        help="LLM port for research queries")
    parser.add_argument("--budget", type=int, default=20,
                        help="Max experiments")
    parser.add_argument("--seed",
                        help="Seed experiment .env to run first")
    parser.add_argument("--memory", default="unknown",
                        help="Total memory (e.g. '128GB')")
    parser.add_argument("--branch", default="",
                        help="Git branch for research (default: autoresearch/{gpu})")

    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()

    # Load bench config
    bench_env = script_dir / "bench.env"
    bench_config = {}
    if bench_env.exists():
        with open(bench_env) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    bench_config[key.strip()] = val.strip()

    gpu = bench_config.get("BENCH_GPU", os.environ.get("BENCH_GPU", "unknown"))
    model = bench_config.get("BENCH_MODEL", os.environ.get("BENCH_MODEL", "unknown"))
    bench_host = bench_config.get("BENCH_REMOTE_HOST",
                                  os.environ.get("BENCH_REMOTE_HOST", ""))
    bench_port = int(bench_config.get("BENCH_SERVER_PORT",
                                      os.environ.get("BENCH_SERVER_PORT", "8099")))

    research_host = args.research_host or bench_host
    research_port = args.research_port or bench_port

    if not research_host:
        print("ERROR: No LLM host. Set --research-host or BENCH_REMOTE_HOST")
        sys.exit(1)

    # Output dirs
    output_dir = script_dir / "output" / gpu / model
    results_dir = output_dir / "results"
    experiments_dir = output_dir / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Results TSV
    tsv_path = output_dir / "results.tsv"
    init_results_tsv(tsv_path)

    # Create research branch
    branch = args.branch or f"autoresearch/{gpu}"
    current = git_current_branch(cwd=str(script_dir))
    if current != branch:
        existing, rc = git_run("branch", "--list", branch, cwd=str(script_dir))
        if existing:
            git_run("checkout", branch, cwd=str(script_dir))
        else:
            git_run("checkout", "-b", branch, cwd=str(script_dir))
        print(f"[*] On branch: {branch}")

    # Build system prompt
    system_msg = build_system_prompt(script_dir, gpu, model, args.memory)

    # Track best score
    best_score = 0.0
    no_improve_count = 0
    step = 0

    # Count existing results
    existing = sorted(results_dir.glob("*.json"))
    for rf in existing:
        result = load_result(rf)
        score = compute_composite_score(result)
        if score > best_score:
            best_score = score
    step = len(existing)

    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH: {model} on {gpu}")
    print(f"  Budget: {args.budget} | Existing: {step} | Best: {best_score:.4f}")
    print(f"  Research LLM: {research_host}:{research_port}")
    print(f"  Branch: {branch}")
    print(f"{'='*60}\n")

    # Run seed if provided and no existing results
    if args.seed and step == 0:
        seed_path = Path(args.seed)
        if not seed_path.is_absolute():
            seed_path = script_dir / seed_path
        print(f"[*] Running seed: {seed_path.name}")

        if run_experiment(script_dir, seed_path):
            # Find the result file
            new_results = sorted(results_dir.glob("*.json"))
            if new_results:
                result = load_result(new_results[-1])
                score = compute_composite_score(result)
                best_score = score
                card = result.get("score_card", {})

                append_results_tsv(tsv_path, {
                    "step": 1,
                    "experiment": result.get("experiment", "seed"),
                    "ctx_size": result["config"].get("ctx_size", ""),
                    "cache_k": result["config"].get("cache_k", ""),
                    "cache_v": result["config"].get("cache_v", ""),
                    "batch": result["config"].get("batch", ""),
                    "ubatch": result["config"].get("ubatch", ""),
                    "parallel": result["config"].get("parallel", ""),
                    "coding_grade": card.get("coding", {}).get("grade", "-"),
                    "agentic_grade": card.get("agentic", {}).get("grade", "-"),
                    "speed_grade": card.get("speed", {}).get("grade", "-"),
                    "context_grade": card.get("context", {}).get("grade", "-"),
                    "gen_tps": card.get("speed", {}).get("avg_gen_tps", 0),
                    "pp_tps": card.get("context", {}).get("avg_pp_tps", 0),
                    "composite_score": score,
                    "kept": "yes",
                    "analysis": "seed experiment",
                })

                # Commit seed
                git_commit(
                    f"autoresearch: seed experiment (score={score:.4f})",
                    [
                        str(tsv_path.relative_to(script_dir)),
                        str(results_dir.relative_to(script_dir)),
                        str(experiments_dir.relative_to(script_dir)),
                    ],
                    cwd=str(script_dir),
                )
                step = 1
                print(f"[*] Seed score: {score:.4f}")
        else:
            print("[!] Seed experiment failed")
            sys.exit(1)

    if step == 0:
        print("[!] No results. Provide --seed to start.")
        sys.exit(1)

    # Main autoresearch loop
    while step < args.budget:
        # Feed results.tsv to LLM
        tsv_content = read_results_tsv(tsv_path)
        user_msg = (
            f"RESULTS SO FAR ({step} of {args.budget} budget):\n\n"
            f"```\n{tsv_content}\n```\n\n"
            f"Current best composite score: {best_score:.4f}\n"
            f"Consecutive experiments without improvement: {no_improve_count}\n\n"
            f"Propose the next experiment."
        )

        print(f"\n[*] Step {step + 1}/{args.budget} — querying research LLM...")

        try:
            response = query_llm(research_host, research_port,
                                 system_msg, user_msg)
            proposal = parse_response(response)
        except Exception as e:
            print(f"[!] LLM error: {e}")
            break

        print(f"  Analysis: {proposal.get('analysis', '')}")
        print(f"  Reasoning: {proposal.get('reasoning', '')}")

        # Check convergence
        if proposal.get("converged"):
            rec = proposal.get("recommendation", {})
            print(f"\n{'='*60}")
            print(f"  CONVERGED after {step} experiments")
            print(f"  Best score: {best_score:.4f}")
            if rec:
                print(f"  Recommended config:")
                for k, v in sorted(rec.items()):
                    if k != "name":
                        print(f"    {k}: {v}")
            print(f"  {proposal.get('analysis', '')}")
            print(f"{'='*60}\n")

            # Commit convergence
            append_results_tsv(tsv_path, {
                "step": step + 1,
                "experiment": "CONVERGED",
                "composite_score": best_score,
                "kept": "final",
                "analysis": proposal.get("analysis", ""),
            })
            git_commit(
                f"autoresearch: converged (score={best_score:.4f})",
                [str(tsv_path.relative_to(script_dir))],
                cwd=str(script_dir),
            )
            break

        # Write and run experiment
        exp_config = proposal["experiment"]
        exp_name = f"{step + 1:03d}_auto_{exp_config.get('name', 'unnamed')}"
        env_file = experiments_dir / f"{exp_name}.env"
        result_file = results_dir / f"{exp_name}.json"

        write_env_file(exp_config, env_file)

        print(f"\n[*] Running: {exp_name}")
        print(f"    ctx={exp_config.get('ctx_size')} "
              f"cache={exp_config.get('cache_type_k')}/{exp_config.get('cache_type_v')} "
              f"batch={exp_config.get('batch_size')}/{exp_config.get('ubatch_size')} "
              f"par={exp_config.get('parallel')}")

        success = run_experiment(script_dir, env_file)
        step += 1

        if success and result_file.exists():
            result = load_result(result_file)
            score = compute_composite_score(result)
            card = result.get("score_card", {})
            improved = score > best_score

            kept = "yes" if improved else "no"
            if improved:
                best_score = score
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(f"[*] Score: {score:.4f} (best: {best_score:.4f}) "
                  f"— {'KEPT' if improved else 'DISCARDED'}")

            append_results_tsv(tsv_path, {
                "step": step,
                "experiment": exp_name,
                "ctx_size": exp_config.get("ctx_size", ""),
                "cache_k": exp_config.get("cache_type_k", ""),
                "cache_v": exp_config.get("cache_type_v", ""),
                "batch": exp_config.get("batch_size", ""),
                "ubatch": exp_config.get("ubatch_size", ""),
                "parallel": exp_config.get("parallel", ""),
                "coding_grade": card.get("coding", {}).get("grade", "-"),
                "agentic_grade": card.get("agentic", {}).get("grade", "-"),
                "speed_grade": card.get("speed", {}).get("grade", "-"),
                "context_grade": card.get("context", {}).get("grade", "-"),
                "gen_tps": card.get("speed", {}).get("avg_gen_tps", 0),
                "pp_tps": card.get("context", {}).get("avg_pp_tps", 0),
                "composite_score": score,
                "kept": kept,
                "analysis": proposal.get("analysis", "")[:200],
            })

            # Commit result
            git_commit(
                f"autoresearch: {exp_name} score={score:.4f} ({'kept' if improved else 'discarded'})",
                [
                    str(tsv_path.relative_to(script_dir)),
                    str(env_file.relative_to(script_dir)),
                    str(result_file.relative_to(script_dir)),
                ],
                cwd=str(script_dir),
            )

            # If not improved, reset the experiment files (keep tsv updated)
            if not improved:
                # We want to keep the TSV row but discard the experiment
                # So we reset the commit but re-add the tsv
                git_reset_last(cwd=str(script_dir))
                git_commit(
                    f"autoresearch: {exp_name} score={score:.4f} (discarded)",
                    [str(tsv_path.relative_to(script_dir))],
                    cwd=str(script_dir),
                )
        else:
            no_improve_count += 1
            append_results_tsv(tsv_path, {
                "step": step,
                "experiment": exp_name,
                "ctx_size": exp_config.get("ctx_size", ""),
                "cache_k": exp_config.get("cache_type_k", ""),
                "cache_v": exp_config.get("cache_type_v", ""),
                "batch": exp_config.get("batch_size", ""),
                "ubatch": exp_config.get("ubatch_size", ""),
                "parallel": exp_config.get("parallel", ""),
                "composite_score": 0,
                "kept": "error",
                "analysis": "experiment failed",
            })
            git_commit(
                f"autoresearch: {exp_name} FAILED",
                [str(tsv_path.relative_to(script_dir))],
                cwd=str(script_dir),
            )
            print(f"[!] Experiment failed — logged to results.tsv")

        # Cool down
        print("[*] Cooling down 10s...")
        time.sleep(10)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  RESEARCH COMPLETE")
    print(f"  Experiments: {step} | Best score: {best_score:.4f}")
    print(f"  Branch: {branch}")
    print(f"  Results: {tsv_path}")
    print(f"{'='*60}\n")

    # Print final comparison
    from evaluate import summarize_all
    summarize_all(str(results_dir))


if __name__ == "__main__":
    main()
