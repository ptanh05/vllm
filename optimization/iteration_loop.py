#!/usr/bin/env python3
"""
Iterative optimization loop:
  benchmark → modify config → benchmark → compare → keep if better.

Usage:
    # Dry-run: compare baseline vs recommended configs without running
    python -m optimization.iteration_loop --trace trace-round1.jsonl --dry-run

    # Full loop with config search:
    python -m optimization.iteration_loop --trace trace-round1.jsonl \\
        --endpoint http://localhost:8000/v1 --model Qwen3.5-2B \\
        --iterations 5 --output-dir iterate_output
"""
import argparse
import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from benchmark.replay_trace import run_replay
from benchmark.metrics import ERSConfig, ERSScorer, final_score
from optimization.config import OptConfig, PRESETS


@dataclass
class IterationResult:
    iteration: int
    config_name: str
    config: dict
    ers: float
    successful: int
    total: int
    mean_ttft_ms: float
    mean_tpot_ms: float
    better_than_baseline: bool | None = None


class Optimizer:
    """Iterative optimiser that benchmarks → tunes → benchmarks again."""

    def __init__(
        self,
        trace_path: str,
        endpoint: str,
        model: str,
        output_dir: str,
        concurrency: int = 8,
        baseline_config: OptConfig | None = None,
    ):
        self.trace_path = trace_path
        self.endpoint = endpoint
        self.model = model
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.baseline_config = baseline_config or OptConfig.baseline()
        self.results: list[IterationResult] = []
        self.best_ers = 0.0
        self.best_config = None

    async def run_iteration(
        self,
        iteration: int,
        config: OptConfig,
        config_name: str,
    ) -> IterationResult:
        """Run one benchmark iteration with the given config."""
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}: {config_name}")
        print(f"{'='*60}")
        for k, v in sorted(config.to_dict().items()):
            if v is not None and v is not False:
                print(f"    {k}: {v}")

        # Run benchmark
        iter_dir = os.path.join(self.output_dir, f"iter_{iteration:03d}_{config_name}")
        os.makedirs(iter_dir, exist_ok=True)

        # Save config
        with open(os.path.join(iter_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        try:
            report = await run_replay(
                trace_path=self.trace_path,
                endpoint=self.endpoint,
                model=self.model,
                concurrency=self.concurrency,
                output_dir=iter_dir,
                respect_arrivals=True,
            )
            ers = report["summary"]["overall_ers"]
            successful = report["summary"]["successful"]
            total = report["summary"]["total_requests"]
            mean_ttft = report["latency"]["ttft_ms"]["mean"]
            mean_tpot = report["latency"]["tpot_mean_ms"]["mean"]
        except Exception as e:
            print(f"  [ERROR] Benchmark failed: {e}")
            ers = 0.0
            successful = 0
            total = 0
            mean_ttft = 999
            mean_tpot = 999

        result = IterationResult(
            iteration=iteration,
            config_name=config_name,
            config=config.to_dict(),
            ers=ers,
            successful=successful,
            total=total,
            mean_ttft_ms=mean_ttft,
            mean_tpot_ms=mean_tpot,
        )

        # Compare to baseline
        if self.results:
            baseline_ers = self.results[0].ers
            result.better_than_baseline = ers > baseline_ers
        else:
            result.better_than_baseline = None

        self.results.append(result)

        if ers > self.best_ers:
            self.best_ers = ers
            self.best_config = copy.deepcopy(config)

        print(f"\n  Result: ERS={ers:.6f}  ({'↑' if result.better_than_baseline else '↓'} vs baseline)"
              f"  TTFT={mean_ttft:.1f}ms  TPOT={mean_tpot:.1f}ms"
              f"  {successful}/{total} successful")

        return result

    def print_leaderboard(self):
        """Print sorted leaderboard of all iterations."""
        print(f"\n{'='*70}")
        print(f"  LEADERBOARD")
        print(f"{'='*70}")
        sorted_results = sorted(self.results, key=lambda r: r.ers, reverse=True)
        print(f"  {'Rank':>4} {'ERS':>10} {'Config':<30} {'TTFT':>8} {'TPOT':>8} {'Succ':>5}")
        print(f"  {'-'*4} {'-'*10} {'-'*30} {'-'*8} {'-'*8} {'-'*5}")
        for rank, r in enumerate(sorted_results):
            marker = "★" if r.better_than_baseline else " "
            print(f"  {rank+1:>4} {r.ers:>10.6f} {marker} {r.config_name:<30s} "
                  f"{r.mean_ttft_ms:>8.1f} {r.mean_tpot_ms:>8.1f} "
                  f"{r.successful:>3}/{r.total:>2}")

        # Save leaderboard
        import csv
        lb_path = os.path.join(self.output_dir, "iteration_leaderboard.csv")
        with open(lb_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "ers", "config_name", "mean_ttft_ms", "mean_tpot_ms",
                        "successful", "total"])
            for rank, r in enumerate(sorted_results):
                w.writerow([rank+1, r.ers, r.config_name, r.mean_ttft_ms,
                           r.mean_tpot_ms, r.successful, r.total])

        # Save best config
        if self.best_config:
            best_path = os.path.join(self.output_dir, "best_config.json")
            with open(best_path, "w") as f:
                json.dump({
                    "best_ers": self.best_ers,
                    "config": self.best_config.to_dict(),
                    "command_args": self.best_config.to_command_args(),
                }, f, indent=2)
            print(f"\n  Best config: ERS={self.best_ers:.6f}")
            print(f"  Saved to: {best_path}")


async def dry_run(trace_path: str, output_dir: str):
    """Show config comparison without running benchmarks."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ITERATIVE OPTIMIZATION — Dry Run")
    print(f"  Trace: {trace_path}")
    print(f"{'='*70}\n")

    configs = [
        ("baseline", OptConfig.baseline()),
        ("recommended", OptConfig.recommended()),
        ("aggressive", OptConfig.aggressive()),
    ]

    # Also generate candidate variations
    configs.append(("baseline+fp8", OptConfig(
        **{**OptConfig.baseline().to_dict(), "kv_cache_dtype": "fp8_e4m3"}
    )))
    configs.append(("baseline+fp8+w8a8", OptConfig(
        **{**OptConfig.baseline().to_dict(), "kv_cache_dtype": "fp8_e4m3",
           "quantization": "fp8"}
    )))
    configs.append(("recommended+fp8+throughput", OptConfig.recommended()))

    print(f"{'Config':<35s} {'KV Cache':<10} {'Quant':<8} {'Block':<6} {'M. Seqs':<8} {'Perf Mode':<12} {'O.Lvl':<5}")
    print(f"{'-'*35} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*12} {'-'*5}")
    for name, cfg in configs:
        print(f"{name:<35s} {cfg.kv_cache_dtype:<10} {str(cfg.quantization or '-'):<8} "
              f"{cfg.block_size:<6} {str(cfg.max_num_seqs or 'auto'):<8} "
              f"{cfg.performance_mode:<12} {cfg.optimization_level:<5}")

    # Save all candidate configs
    candidates = []
    for name, cfg in configs:
        candidates.append({"name": name, **cfg.to_dict()})
    cand_path = os.path.join(output_dir, "candidate_configs.json")
    with open(cand_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\nCandidate configs saved to {cand_path}")

    print(f"\n{'='*70}")
    print(f"  Recommended iteration order:")
    print(f"    1. baseline               → baseline ERS")
    print(f"    2. baseline+fp8           → measure KV cache FP8 impact")
    print(f"    3. baseline+fp8+w8a8      → measure weight FP8 impact")
    print(f"    4. recommended            → combined optimization")
    print(f"    5. aggressive             → push limits")
    print(f"    6. ... fine-tune based on profiling")
    print(f"{'='*70}")


async def main():
    p = argparse.ArgumentParser(description="Iterative Optimization Loop")
    p.add_argument("--trace", required=True, help="trace-round1.jsonl")
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--output-dir", default="iterate_output")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        await dry_run(args.trace, args.output_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Save run metadata
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "trace": args.trace,
            "model": args.model,
            "iterations": args.iterations,
        }, f, indent=2)

    opt = Optimizer(
        trace_path=args.trace,
        endpoint=args.endpoint,
        model=args.model,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
    )

    # Define iteration sequence
    iterations = [
        ("baseline", OptConfig.baseline()),
        ("baseline+fp8_kv", OptConfig(
            kv_cache_dtype="fp8_e4m3",
        )),
        ("baseline+fp8_kv+fp8_weight", OptConfig(
            kv_cache_dtype="fp8_e4m3",
            quantization="fp8",
        )),
        ("recommended", OptConfig.recommended()),
        ("aggressive", OptConfig.aggressive()),
    ]

    # Add additional iterations if requested
    while len(iterations) < args.iterations:
        # Generate variants by toggling parameters
        prev_name, prev_cfg = iterations[-1]
        variants = [
            ("max_seqs_half", "max_num_seqs",
             max(64, (prev_cfg.max_num_seqs or 256) // 2)),
            ("max_seqs_double", "max_num_seqs",
             min(2048, (prev_cfg.max_num_seqs or 256) * 2)),
            ("block_64", "block_size", 64),
            ("block_16", "block_size", 16),
            ("perf_balanced", "performance_mode", "balanced"),
        ]
        added = False
        for vname, vkey, vval in variants:
            if len(iterations) >= args.iterations:
                break
            full_name = f"{prev_name}+{vname}"
            # Don't duplicate existing configs
            if any(full_name == r.config_name for r in opt.results):
                continue
            new_cfg = copy.deepcopy(prev_cfg)
            setattr(new_cfg, vkey, vval)
            iterations.append((full_name, new_cfg))
            added = True
        if not added:
            break  # no more unique variants

    # Run iterations
    for i, (name, cfg) in enumerate(iterations[:args.iterations]):
        # For non-baseline iterations, inherit baseline defaults for unspecified fields
        if name != "baseline":
            base = OptConfig.baseline().to_dict()
            override = {k: v for k, v in cfg.to_dict().items() if v is not None}
            base.update(override)
            cfg = OptConfig(**base)

        result = await opt.run_iteration(i, cfg, name)

        # Save intermediate leaderboard
        opt.print_leaderboard()

    # Final leaderboard
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    opt.print_leaderboard()

    improvement = (
        (opt.best_ers - opt.results[0].ers) / max(opt.results[0].ers, 0.0001) * 100
        if opt.results else 0
    )
    print(f"\n  Improvement over baseline: {improvement:+.1f}%")
    print(f"  Best ERS: {opt.best_ers:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
