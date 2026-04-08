from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_eval_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def collect_task_stats(eval_root: Path) -> dict[str, dict[str, float | int]]:
    video_root = eval_root / "video"
    task_root = video_root if video_root.exists() else eval_root

    task_stats: dict[str, dict[str, float | int]] = {}

    for task_dir in sorted(task_root.iterdir()):
        if not task_dir.is_dir():
            continue

        videos = [p for p in task_dir.rglob("*") if is_eval_video(p)]
        total = len(videos)
        success = sum(1 for p in videos if "_success" in p.stem)
        rate = (success / total) if total > 0 else 0.0

        task_stats[task_dir.name] = {
            "total": total,
            "success": success,
            "success_rate": rate,
        }

    return task_stats


def print_stats(task_stats: dict[str, dict[str, float | int]]) -> None:
    print("Per-task success rate:")
    print("-" * 60)

    total_all = 0
    success_all = 0

    for task, stats in task_stats.items():
        total = int(stats["total"])
        success = int(stats["success"])
        rate = float(stats["success_rate"])
        total_all += total
        success_all += success

        print(f"{task:<20} {success:>4}/{total:<4}  ({rate:.2%})")

    overall_rate = (success_all / total_all) if total_all > 0 else 0.0
    print("-" * 60)
    print(f"{'Overall':<20} {success_all:>4}/{total_all:<4}  ({overall_rate:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count success rates per task from eval videos."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_result",
        help="Path to eval_result directory (default: ./eval_result).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output json path for stats.",
    )
    args = parser.parse_args()

    if not args.eval_root.exists():
        raise FileNotFoundError(f"Directory not found: {args.eval_root}")

    task_stats = collect_task_stats(args.eval_root)
    print_stats(task_stats)

    if args.save_json is not None:
        total_all = sum(int(v["total"]) for v in task_stats.values())
        success_all = sum(int(v["success"]) for v in task_stats.values())
        output = {
            "overall": {
                "total": total_all,
                "success": success_all,
                "success_rate": (success_all / total_all) if total_all > 0 else 0.0,
            },
            "per_task": task_stats,
        }
        args.save_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nSaved stats to: {args.save_json}")


if __name__ == "__main__":
    main()
