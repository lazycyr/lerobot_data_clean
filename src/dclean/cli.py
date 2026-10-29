"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import run_check
from .clean import (
    format_dry_run,
    plan_clean,
    prepare_clean_output,
    run_clean,
    validate_clean_paths,
)
from .config import load_config
from .dataset import video_cameras
from .progress import Reporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dclean",
        description="Read-only quality check for LeRobot Dataset v3.0.",
    )
    parser.add_argument("--version", action="version", version=f"dclean {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check a LeRobot v3.0 dataset (read-only)")
    check.add_argument("--dataset", type=Path, required=True, help="dataset root")
    check.add_argument("--out", type=Path, required=True, help="output directory")
    check.add_argument(
        "--episodes",
        default=None,
        help="episode selection, e.g. 0-9,15,20-22 (default: all)",
    )
    check.add_argument("--config", type=Path, default=None, help="TOML threshold config")
    check.add_argument("--no-video", action="store_true", help="skip ffprobe/video checks")
    check.add_argument(
        "--no-video-quality",
        action="store_true",
        help="skip sampled frame-quality checks (black/frozen/blur/exposure)",
    )
    check.add_argument("--no-progress", action="store_true", help="disable progress bars")
    check.add_argument(
        "--full-report",
        action="store_true",
        help="also write all-episode tables, calibration JSON and JSON copies",
    )
    check.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into a non-empty output directory",
    )

    clean = sub.add_parser(
        "clean",
        help="delete episodes and write a new, re-indexed dataset copy",
    )
    clean.add_argument("--dataset", type=Path, required=True, help="source dataset root (read-only)")
    clean.add_argument("--out", type=Path, required=True, help="new dataset directory")
    clean.add_argument(
        "--episodes",
        required=True,
        help="episodes to delete, e.g. 32,88,91,115 (ranges allowed: 10-12)",
    )
    clean.add_argument(
        "--check-report",
        type=Path,
        default=None,
        help="optional episode_quality.csv used to copy deletion reasons into the report",
    )
    clean.add_argument(
        "--vcodec",
        default="libx264",
        help="ffmpeg video encoder: libx264 (h264, default) or libsvtav1 (av1)",
    )
    clean.add_argument(
        "--dry-run",
        action="store_true",
        help="only print the deletion plan and mapping; write nothing",
    )
    clean.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into a non-empty output directory (removes old meta/data/videos)",
    )
    clean.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return parser


def _prepare_output(dataset: Path, out: Path, overwrite: bool) -> tuple[Path, Path]:
    dataset = dataset.resolve()
    if not (dataset / "meta" / "info.json").is_file():
        raise ValueError(f"not a LeRobot dataset (missing meta/info.json): {dataset}")
    out = out.resolve()
    if out == dataset or dataset in out.parents:
        raise ValueError("--out must not be inside --dataset")
    if out.exists():
        if not out.is_dir():
            raise ValueError(f"--out exists and is not a directory: {out}")
        if any(out.iterdir()) and not overwrite:
            raise ValueError(f"output directory is not empty (use --overwrite): {out}")
    else:
        out.mkdir(parents=True, exist_ok=True)
    return dataset, out


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        dataset, out = _prepare_output(args.dataset, args.out, args.overwrite)
        cfg = load_config(args.config)
    except (ValueError, OSError) as exc:
        print(f"dclean: error: {exc}", file=sys.stderr)
        return 2

    reporter = Reporter(total_stages=7, enabled=not args.no_progress)
    try:
        result = run_check(
            dataset,
            out,
            args.episodes,
            cfg,
            no_video=args.no_video,
            no_video_quality=args.no_video_quality,
            full_report=args.full_report,
            reporter=reporter,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should report, not traceback
        print(f"dclean: fatal: {exc}", file=sys.stderr)
        return 1

    fail_rows = sum(1 for issue in result.episode_issues if issue.severity == "fail")
    warn_rows = sum(1 for issue in result.episode_issues if issue.severity == "warn")
    print(
        f"dclean check complete: {len(result.episode_issues)} issue row(s), "
        f"{fail_rows} fail, {warn_rows} warn"
    )
    print(f"  episodes: {result.quality_csv_path}")
    if result.quality_json_path is not None:
        print(f"  json    : {result.quality_json_path}")
    if result.trajectory_scores_path is not None:
        print(f"  traj    : {result.trajectory_scores_path}")
        print(f"  calib   : {result.trajectory_calibration_path}")
    if result.video_quality_path is not None:
        print(f"  video   : {result.video_quality_path}")
    if result.av_sync_path is not None:
        print(f"  avsync  : {result.av_sync_path}")
    print(f"  summary : {result.summary_path}")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    try:
        dataset, out = validate_clean_paths(args.dataset, args.out)
        plan = plan_clean(
            dataset,
            args.episodes,
            check_report=args.check_report,
            vcodec=args.vcodec,
        )
    except (ValueError, OSError, FileNotFoundError) as exc:
        print(f"dclean: error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(format_dry_run(plan), end="")
        return 0

    try:
        prepare_clean_output(out, args.overwrite, video_cameras(plan.info))
    except (ValueError, OSError) as exc:
        print(f"dclean: error: {exc}", file=sys.stderr)
        return 2

    reporter = Reporter(total_stages=6, enabled=not args.no_progress)
    try:
        result = run_clean(plan, out, reporter=reporter, prepare=False)
    except Exception as exc:  # noqa: BLE001 - CLI should report, not traceback
        print(f"dclean: fatal: {exc}", file=sys.stderr)
        return 1

    print(
        f"dclean clean complete: deleted {result.deleted_count}, kept {result.kept_count}, "
        f"{result.total_frames} frames"
    )
    print(f"  dataset : {result.out}")
    print(f"  report  : {result.report_path}")
    print(f"  deleted : {result.deleted_csv}")
    print(f"  mapping : {result.mapping_csv}")
    if result.verification_issues:
        print(
            f"dclean: verification found {len(result.verification_issues)} issue(s); "
            "see the report",
            file=sys.stderr,
        )
        for text in result.verification_issues[:10]:
            print(f"  - {text}", file=sys.stderr)
        if len(result.verification_issues) > 10:
            print(
                f"  ... {len(result.verification_issues) - 10} more",
                file=sys.stderr,
            )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "clean":
        return _cmd_clean(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
