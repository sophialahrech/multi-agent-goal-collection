from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from dg_commons.sim.simulator import Simulator
from pdm4ar.exercises_def.ex14.perf_metrics import PlayerMetrics, ex14_metrics
from pdm4ar.exercises_def.ex14.utils_config import load_config, sim_context_from_config

DEFAULT_CONFIGS = [
    "src/pdm4ar/exercises_def/ex14/config_1.yaml",
    "src/pdm4ar/exercises_def/ex14/config_2.yaml",
    "src/pdm4ar/exercises_def/ex14/config_3.yaml",
]


def _player_metrics_to_dict(metrics: Iterable[PlayerMetrics]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pm in metrics:
        out.append(
            {
                "player_name": str(pm.player_name),
                "collided": pm.collided,
                "num_goal_delivered": pm.num_goal_delivered,
                "travelled_distance": pm.travelled_distance,
                "actuation_effort": pm.actuation_effort,
                "avg_computation_time": pm.avg_computation_time,
            }
        )
    return out


def _extract_collision_names(sim_context) -> List[str]:
    collided: set[str] = set()
    for cr in sim_context.collision_reports:
        collided.update(cr.players.keys())
    return sorted(collided)


def _extract_last_events(sim_context) -> Dict[str, Any]:
    events: Dict[str, Any] = {}
    for name, log in sim_context.log.items():
        extra_values = getattr(log.extra, "values", [])
        if not extra_values:
            continue
        last = extra_values[-1]
        if isinstance(last, Mapping):
            events[name] = {k: v for k, v in last.items() if k in ("events", "mode", "target")}
    return events


def run_episode(config_path: str, seed: int | None = None) -> Dict[str, Any]:
    cfg = dict(load_config(config_path))
    if seed is not None:
        cfg["seed"] = int(seed)
    sim_context = sim_context_from_config(cfg)
    Simulator().run(sim_context)
    agg_metrics, player_metrics = ex14_metrics(sim_context)
    result = {
        "config": cfg.get("config_name"),
        "seed": cfg.get("seed"),
        "score": agg_metrics.reduce_to_score(),
        "num_goals_delivered": agg_metrics.num_goals_delivered,
        "num_collided_players": agg_metrics.num_collided_players,
        "task_time": agg_metrics.task_accomplishment_time,
        "travelled_distance": agg_metrics.total_travelled_distance,
        "actuation_effort": agg_metrics.total_actuation_effort,
        "avg_computation_time": agg_metrics.avg_computation_time,
        "global_planning_time": agg_metrics.global_planning_time,
        "collided_players": _extract_collision_names(sim_context),
        "players": _player_metrics_to_dict(player_metrics),
        "events": _extract_last_events(sim_context),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch runner for ex14.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="List of config file paths to run.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of seeds per config.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset added to config seeds.")
    parser.add_argument(
        "--output",
        type=str,
        default="out/ex14_batch.jsonl",
        help="Path to JSONL log file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        for config_path in args.configs:
            for run_idx in range(args.runs):
                seed = args.seed_offset + run_idx
                entry = run_episode(config_path, seed=seed)
                fp.write(json.dumps(entry) + "\n")
                print(f"Ran {config_path} seed={seed}: score={entry['score']:.2f}")


if __name__ == "__main__":
    main()
