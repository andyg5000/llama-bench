#!/usr/bin/env python3
"""Evaluate llama-server performance across use-case profiles.

Profiles:
  coding   - Code comprehension, editing, generation, debugging
  agentic  - Tool calling, structured output, multi-turn, instruction adherence
  speed    - Quick accuracy checks + generation/prompt processing benchmarks
  context  - Needle-in-a-haystack at multiple context lengths

Scores each profile and produces a score card for comparing configs.
"""
import argparse
import concurrent.futures
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def query_server(host: str, port: int, prompt: str,
                 max_tokens: int = 500, timeout: int = 300) -> dict:
    """Send a chat completion request and measure timing."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e), "elapsed": time.perf_counter() - start}
    elapsed = time.perf_counter() - start

    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    raw_content = choice.get("message", {}).get("content", "").strip()

    return {
        "content": strip_thinking(raw_content),
        "raw_content": raw_content,
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "server_prompt_tps": timings.get("prompt_per_second", 0),
        "server_gen_tps": timings.get("predicted_per_second", 0),
        "server_prompt_ms": timings.get("prompt_ms", 0),
        "server_gen_ms": timings.get("predicted_ms", 0),
        "predicted_n": timings.get("predicted_n", 0),
        "has_thinking": "<think>" in raw_content,
    }


def get_server_metrics(host: str, port: int) -> dict:
    """Get server-side performance metrics."""
    url = f"http://{host}:{port}/metrics"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode()
    except Exception:
        return {}

    metrics = {}
    for line in text.split("\n"):
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return metrics


def check_answer(response: str, expected: str, match_type: str) -> bool:
    """Check if response matches expected answer."""
    response_clean = response.strip().lower().rstrip(".")
    expected_clean = expected.strip().lower()
    return expected_clean in response_clean


def grade(score: float) -> str:
    """Convert a 0-1 score to a letter grade."""
    if score >= 0.95:
        return "A+"
    if score >= 0.90:
        return "A"
    if score >= 0.85:
        return "A-"
    if score >= 0.80:
        return "B+"
    if score >= 0.75:
        return "B"
    if score >= 0.70:
        return "B-"
    if score >= 0.60:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Suite runners
# ---------------------------------------------------------------------------

def run_prompt_suite(host: str, port: int, prompts_file: Path,
                     max_tokens: int = 4000) -> dict:
    """Run a JSONL prompt suite and return results."""
    results = []
    correct = 0
    total = 0
    gen_tps_vals = []
    elapsed_vals = []

    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test = json.loads(line)
            test_id = test["id"]
            print(f"  [{test_id}] ", end="", flush=True)

            # For non-thinking prompts, append /no_think to reduce overhead
            prompt = test["prompt"]
            if test.get("thinking") is False:
                prompt = prompt + " /no_think"

            resp = query_server(host, port, prompt, max_tokens)
            if "error" in resp:
                print(f"ERROR: {resp['error']}")
                results.append({"id": test_id, "error": resp["error"]})
                total += 1
                continue

            is_correct = check_answer(
                resp["content"], test["expected"], test.get("match", "contains")
            )
            if is_correct:
                correct += 1
            total += 1

            gen_tps = resp.get("server_gen_tps", 0)
            if gen_tps > 0:
                gen_tps_vals.append(gen_tps)
            elapsed_vals.append(resp["elapsed"])

            results.append({
                "id": test_id,
                "category": test.get("category", ""),
                "prompt": test["prompt"][:100],
                "expected": test["expected"],
                "response": resp["content"][:300],
                "correct": is_correct,
                "elapsed": round(resp["elapsed"], 3),
                "completion_tokens": resp["completion_tokens"],
                "predicted_n": resp.get("predicted_n", 0),
                "server_gen_tps": round(gen_tps, 2),
                "server_prompt_tps": round(resp.get("server_prompt_tps", 0), 2),
                "has_thinking": resp.get("has_thinking", False),
            })

            status = "OK" if is_correct else "FAIL"
            print(f"{status} ({resp.get('predicted_n', 0)}tok, "
                  f"{gen_tps:.1f} t/s, {resp['elapsed']:.2f}s)")

    accuracy = correct / total if total > 0 else 0
    avg_gen_tps = statistics.mean(gen_tps_vals) if gen_tps_vals else 0
    avg_elapsed = statistics.mean(elapsed_vals) if elapsed_vals else 0

    return {
        "tests": results,
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": total,
        "avg_gen_tps": round(avg_gen_tps, 2),
        "avg_elapsed": round(avg_elapsed, 3),
    }


def run_context_suite(host: str, port: int, prompts_file: Path) -> dict:
    """Run needle-in-a-haystack and generation benchmarks."""
    results = []
    pp_tps_vals = []

    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test = json.loads(line)
            test_id = test["id"]
            print(f"  [{test_id}] ", end="", flush=True)

            if test.get("prompt_type") == "filler_needle":
                filler = test["filler"] * test["filler_repeats"]
                # Insert needle at ~40% through the text
                insert_pos = len(filler) * 2 // 5
                prompt = (
                    f"Read the following text carefully and answer the "
                    f"question at the end.\n\n"
                    f"{filler[:insert_pos]}{test['needle']} "
                    f"{filler[insert_pos:]}\n\n"
                    f"Question: {test['question']} /no_think"
                )
                max_tok = 500
            elif test.get("prompt_type") == "generation":
                prompt = test["prompt"]
                max_tok = test.get("max_tokens", 2000)
            else:
                continue

            timeout = max(300, test.get("filler_repeats", 0) // 10 + 300)
            resp = query_server(host, port, prompt, max_tok, timeout=timeout)

            if "error" in resp:
                print(f"ERROR: {resp['error']}")
                results.append({"id": test_id, "error": resp["error"]})
                continue

            entry = {
                "id": test_id,
                "category": test.get("category", ""),
                "description": test.get("description", ""),
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "server_prompt_tps": round(resp.get("server_prompt_tps", 0), 2),
                "server_gen_tps": round(resp.get("server_gen_tps", 0), 2),
                "elapsed": round(resp["elapsed"], 3),
            }

            if test.get("prompt_type") == "filler_needle":
                is_correct = check_answer(
                    resp["content"], test["expected"], test.get("match", "contains")
                )
                entry["correct"] = is_correct
                entry["response"] = resp["content"][:200]
                pp_tps = resp.get("server_prompt_tps", 0)
                if pp_tps > 0:
                    pp_tps_vals.append(pp_tps)
                status = "OK" if is_correct else "FAIL"
                print(f"{status} (pp: {pp_tps:.0f} t/s, "
                      f"{resp['prompt_tokens']} prompt tok)")
            else:
                entry["predicted_n"] = resp.get("predicted_n", 0)
                print(f"{resp.get('predicted_n', 0)}tok, "
                      f"{resp.get('server_gen_tps', 0):.1f} t/s, "
                      f"{resp['elapsed']:.2f}s")

            results.append(entry)

    return {
        "tests": results,
        "avg_pp_tps": round(statistics.mean(pp_tps_vals), 2) if pp_tps_vals else 0,
    }


def run_concurrent_test(host: str, port: int, num_requests: int = 4) -> dict:
    """Send multiple requests concurrently to test parallel throughput."""
    prompts = [
        "What is the capital of Japan? Only respond with the city name.",
        "What is 42 * 17? Only respond with the number.",
        "Name the largest planet in our solar system. Only one word.",
        "What programming language is known for the GIL? Only one word.",
    ]
    expected = ["tokyo", "714", "jupiter", "python"]

    print(f"  [concurrent_{num_requests}x] ", end="", flush=True)

    start = time.perf_counter()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as pool:
        futures = {
            pool.submit(query_server, host, port, p, 500): (p, e)
            for p, e in zip(prompts[:num_requests], expected[:num_requests])
        }
        for future in concurrent.futures.as_completed(futures):
            prompt, exp = futures[future]
            resp = future.result()
            results.append({
                "prompt": prompt[:60],
                "expected": exp,
                "correct": check_answer(
                    resp.get("content", ""), exp, "contains"
                ) if "error" not in resp else False,
                "elapsed": round(resp.get("elapsed", 0), 3),
                "server_gen_tps": round(resp.get("server_gen_tps", 0), 2),
                "error": resp.get("error"),
            })

    wall_time = time.perf_counter() - start
    correct = sum(1 for r in results if r.get("correct"))
    total_tokens = sum(r.get("server_gen_tps", 0) for r in results)

    print(f"{correct}/{num_requests} correct, "
          f"{wall_time:.2f}s wall, "
          f"~{total_tokens:.0f} aggregate t/s")

    return {
        "num_requests": num_requests,
        "wall_time": round(wall_time, 3),
        "results": results,
        "correct": correct,
        "aggregate_tps": round(total_tokens, 2),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    """Run the full evaluation suite across all profiles."""
    prompts_dir = Path(args.prompts_dir)
    host, port = args.host, args.port

    results = {
        "experiment": args.experiment,
        "gpu": args.gpu,
        "model": args.model,
        "run": args.run,
        "config": {
            "ctx_size": args.ctx_size,
            "cache_k": args.cache_k,
            "cache_v": args.cache_v,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "parallel": args.parallel,
        },
        "profiles": {},
        "score_card": {},
    }

    # --- Coding profile ---
    coding_file = prompts_dir / "coding.jsonl"
    if coding_file.exists():
        print("\n=== CODING PROFILE ===")
        results["profiles"]["coding"] = run_prompt_suite(
            host, port, coding_file, max_tokens=2000
        )

    # --- Agentic profile ---
    agentic_file = prompts_dir / "agentic.jsonl"
    if agentic_file.exists():
        print("\n=== AGENTIC PROFILE ===")
        agentic = run_prompt_suite(host, port, agentic_file, max_tokens=2000)

        # Concurrent throughput test
        print("\n--- Concurrent load ---")
        agentic["concurrent"] = run_concurrent_test(host, port, num_requests=4)
        results["profiles"]["agentic"] = agentic

    # --- Speed profile ---
    speed_file = prompts_dir / "speed.jsonl"
    if speed_file.exists():
        print("\n=== SPEED PROFILE ===")
        results["profiles"]["speed"] = run_prompt_suite(
            host, port, speed_file, max_tokens=2000
        )

    # --- Context profile ---
    context_file = prompts_dir / "long_context.jsonl"
    if context_file.exists():
        print("\n=== CONTEXT PROFILE ===")
        results["profiles"]["context"] = run_context_suite(
            host, port, context_file
        )

    # --- Server metrics / memory ---
    metrics = get_server_metrics(host, port)
    results["server_metrics"] = {
        k: v for k, v in metrics.items()
        if any(x in k for x in [
            "prompt_tokens", "generation", "kv_cache",
            "requests", "memory", "vram"
        ])
    }

    # --- Score card ---
    results["score_card"] = build_score_card(results)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print_score_card(results)
    return results


def build_score_card(results: dict) -> dict:
    """Build per-profile scores from results."""
    card = {}
    profiles = results.get("profiles", {})

    # Coding: accuracy + avg response speed
    if "coding" in profiles:
        p = profiles["coding"]
        card["coding"] = {
            "accuracy": p["accuracy"],
            "avg_gen_tps": p["avg_gen_tps"],
            "avg_elapsed": p["avg_elapsed"],
            "grade": grade(p["accuracy"]),
        }

    # Agentic: accuracy + concurrent throughput
    if "agentic" in profiles:
        p = profiles["agentic"]
        conc = p.get("concurrent", {})
        conc_score = conc.get("correct", 0) / max(conc.get("num_requests", 1), 1)
        combined = (p["accuracy"] * 0.7) + (conc_score * 0.3)
        card["agentic"] = {
            "accuracy": p["accuracy"],
            "concurrent_accuracy": round(conc_score, 3),
            "concurrent_wall_time": conc.get("wall_time", 0),
            "combined_score": round(combined, 3),
            "grade": grade(combined),
        }

    # Speed: accuracy + gen tps
    if "speed" in profiles:
        p = profiles["speed"]
        card["speed"] = {
            "accuracy": p["accuracy"],
            "avg_gen_tps": p["avg_gen_tps"],
            "grade": grade(p["accuracy"]),
        }

    # Context: needle accuracy across lengths + pp speed
    if "context" in profiles:
        p = profiles["context"]
        needle_tests = [t for t in p.get("tests", [])
                        if "correct" in t]
        needle_correct = sum(1 for t in needle_tests if t["correct"])
        needle_total = len(needle_tests)
        needle_acc = needle_correct / needle_total if needle_total > 0 else 0
        card["context"] = {
            "needle_accuracy": round(needle_acc, 3),
            "needle_correct": needle_correct,
            "needle_total": needle_total,
            "avg_pp_tps": p.get("avg_pp_tps", 0),
            "grade": grade(needle_acc),
        }

    return card


def print_score_card(results: dict):
    """Print a formatted score card."""
    card = results.get("score_card", {})
    config = results.get("config", {})

    print(f"\n{'='*65}")
    print(f"  SCORE CARD: {results.get('experiment', '?')}")
    print(f"  GPU: {results.get('gpu', '?')} | Model: {results.get('model', '?')}")
    if results.get("run"):
        print(f"  Run: {results['run']}")
    print(f"  Config: ctx={config.get('ctx_size')} "
          f"cache={config.get('cache_k')}/{config.get('cache_v')} "
          f"batch={config.get('batch')}/{config.get('ubatch')} "
          f"par={config.get('parallel')}")
    print(f"{'='*65}")

    if "coding" in card:
        c = card["coding"]
        print(f"  Coding:   {c['grade']:>3}  "
              f"(acc={c['accuracy']*100:.0f}% "
              f"gen={c['avg_gen_tps']:.0f} t/s "
              f"avg={c['avg_elapsed']:.1f}s)")

    if "agentic" in card:
        c = card["agentic"]
        print(f"  Agentic:  {c['grade']:>3}  "
              f"(acc={c['accuracy']*100:.0f}% "
              f"conc={c['concurrent_accuracy']*100:.0f}% "
              f"wall={c['concurrent_wall_time']:.1f}s)")

    if "speed" in card:
        c = card["speed"]
        print(f"  Speed:    {c['grade']:>3}  "
              f"(acc={c['accuracy']*100:.0f}% "
              f"gen={c['avg_gen_tps']:.0f} t/s)")

    if "context" in card:
        c = card["context"]
        print(f"  Context:  {c['grade']:>3}  "
              f"(needle={c['needle_correct']}/{c['needle_total']} "
              f"pp={c['avg_pp_tps']:.0f} t/s)")

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def summarize_result(path: str):
    """Print a summary of a single result file."""
    with open(path) as f:
        data = json.load(f)
    print_score_card(data)


def summarize_all(results_dir: str):
    """Print a comparison table of all results."""
    results_path = Path(results_dir)
    files = sorted(results_path.glob("*.json"))
    if not files:
        print("No results found.")
        return

    # Header
    print(f"\n{'Experiment':<30} {'Ctx':>6} {'Cache':>7} "
          f"{'Batch':>10} {'Par':>3} "
          f"{'Coding':>6} {'Agent':>6} {'Speed':>6} {'Ctx':>6} "
          f"{'Gen t/s':>7} {'PP t/s':>7}")
    print("-" * 110)

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        c = data.get("config", {})
        card = data.get("score_card", {})

        coding_g = card.get("coding", {}).get("grade", "-")
        agent_g = card.get("agentic", {}).get("grade", "-")
        speed_g = card.get("speed", {}).get("grade", "-")
        ctx_g = card.get("context", {}).get("grade", "-")
        gen_tps = card.get("speed", {}).get("avg_gen_tps", 0)
        pp_tps = card.get("context", {}).get("avg_pp_tps", 0)

        print(f"{data.get('experiment', '?'):<30} "
              f"{c.get('ctx_size', ''):>6} "
              f"{c.get('cache_k', '')}/{c.get('cache_v', ''):>4} "
              f"{c.get('batch', '')}/{c.get('ubatch', ''):>7} "
              f"{c.get('parallel', ''):>3} "
              f"{coding_g:>6} {agent_g:>6} {speed_g:>6} {ctx_g:>6} "
              f"{gen_tps:>7.1f} {pp_tps:>7.1f}")

    print()


def summarize_runs(results_dir: str, experiment: str):
    """Aggregate multiple runs of the same experiment (mean/stddev)."""
    results_path = Path(results_dir)
    pattern = f"{experiment}_run*.json"
    files = sorted(results_path.glob(pattern))
    if not files:
        print(f"No runs found matching {pattern}")
        return

    gen_tps = []
    pp_tps = []
    accuracies = {"coding": [], "agentic": [], "speed": [], "context": []}

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        card = data.get("score_card", {})
        if "speed" in card:
            gen_tps.append(card["speed"].get("avg_gen_tps", 0))
        if "context" in card:
            pp_tps.append(card["context"].get("avg_pp_tps", 0))
        for profile in accuracies:
            if profile in card:
                acc_key = "accuracy" if profile != "context" else "needle_accuracy"
                accuracies[profile].append(card[profile].get(acc_key, 0))

    print(f"\n{'='*50}")
    print(f"  Aggregate: {experiment} ({len(files)} runs)")
    print(f"{'='*50}")

    if gen_tps:
        m, s = statistics.mean(gen_tps), statistics.stdev(gen_tps) if len(gen_tps) > 1 else 0
        print(f"  Gen TPS:  {m:.1f} +/- {s:.1f}")
    if pp_tps:
        m, s = statistics.mean(pp_tps), statistics.stdev(pp_tps) if len(pp_tps) > 1 else 0
        print(f"  PP TPS:   {m:.1f} +/- {s:.1f}")
    for profile, vals in accuracies.items():
        if vals:
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {profile.capitalize():>8} acc: {m*100:.1f}% +/- {s*100:.1f}%")

    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate llama-server across use-case profiles"
    )
    parser.add_argument("--host", default="", help="Remote server host")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--prompts-dir", default="eval_prompts",
                        help="Directory containing suite JSONL files")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--experiment", default="unnamed")
    parser.add_argument("--gpu", default="unknown")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--run", default="", help="Run number for multi-run")
    parser.add_argument("--ctx-size", default="0")
    parser.add_argument("--cache-k", default="f16")
    parser.add_argument("--cache-v", default="f16")
    parser.add_argument("--batch", default="2048")
    parser.add_argument("--ubatch", default="512")
    parser.add_argument("--parallel", default="1")

    parser.add_argument("--summarize", help="Summarize a single result file")
    parser.add_argument("--summarize-all", help="Summarize all results in dir")
    parser.add_argument("--summarize-runs", nargs=2,
                        metavar=("DIR", "EXPERIMENT"),
                        help="Aggregate multiple runs")

    args = parser.parse_args()

    if args.summarize:
        summarize_result(args.summarize)
    elif args.summarize_all:
        summarize_all(args.summarize_all)
    elif args.summarize_runs:
        summarize_runs(args.summarize_runs[0], args.summarize_runs[1])
    else:
        if not args.host:
            parser.error("--host is required for evaluation")
        if not args.output:
            parser.error("--output is required for evaluation")
        run_evaluation(args)
