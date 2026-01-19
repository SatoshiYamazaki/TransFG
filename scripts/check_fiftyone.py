#!/usr/bin/env python3
"""Headless FiftyOne readiness check for macOS arm64 (MPS/CPU).

This script verifies that FiftyOne can be imported, that the app service can
start in headless mode, and optionally that a COCO validation directory can be
loaded. It prints a JSON report and exits non-zero on failure.

Usage:
    python scripts/check_fiftyone.py
    python scripts/check_fiftyone.py --coco-dir /Users/.../fiftyone/coco-2017/validation
"""
import argparse
import json
import platform
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FiftyOne readiness check")
    parser.add_argument(
        "--coco-dir",
        type=Path,
        default=None,
        help="Optional COCO validation directory to load (e.g., /Users/.../fiftyone/coco-2017/validation)",
    )
    return parser.parse_args()


def detect_coco_paths(base: Path) -> tuple[Path, Path]:
    labels_candidates = [
        base / "annotations" / "instances_val2017.json",
        base / "instances_val2017.json",
    ]
    data_candidates = [
        base / "val2017",
        base / "images",
        base,
    ]

    labels_path = next((p for p in labels_candidates if p.exists()), None)
    data_path = next((p for p in data_candidates if p.exists()), None)
    if labels_path is None or data_path is None:
        missing = []
        if labels_path is None:
            missing.append("instances_val2017.json")
        if data_path is None:
            missing.append("images/val2017")
        raise FileNotFoundError(f"Missing COCO components: {', '.join(missing)} under {base}")
    return data_path, labels_path


def main() -> None:
    args = parse_args()
    report = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    try:
        import fiftyone as fo  # type: ignore
    except Exception as exc:  # pragma: no cover - env probe
        report["status"] = "import_failed"
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        sys.exit(1)

    report.update(
        {
            "status": "import_ok",
            "fiftyone_version": getattr(fo, "__version__", "unknown"),
            "config_path": str(Path(getattr(fo.config, "config_path", ""))),
            "database_dir": str(Path(getattr(fo.config, "database_dir", ""))),
            "fohome": str(Path(getattr(fo.config, "fohome", ""))),
            "use_gpu": bool(getattr(fo.config, "default_ml_backend", "")),
        }
    )

    session = None
    try:
        session = fo.launch_app(auto=False)  # no browser
        report["app_launch"] = "ok"
    except Exception as exc:  # pragma: no cover - env probe
        report["status"] = "app_launch_failed"
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        sys.exit(1)
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    if args.coco_dir is not None:
        coco_report = {"path": str(args.coco_dir)}
        if not args.coco_dir.exists():
            coco_report["status"] = "missing_dir"
            report["status"] = "coco_check_failed"
            report["coco_validation"] = coco_report
            print(json.dumps(report, indent=2))
            sys.exit(1)
        try:
            data_path, labels_path = detect_coco_paths(args.coco_dir)
            ds = fo.Dataset.from_dir(
                name="coco_val_probe",
                dataset_type=fo.types.COCODetectionDataset,
                data_path=str(data_path),
                labels_path=str(labels_path),
                include_id=True,
                persistent=False,
            )
            sample = ds.first() if len(ds) else None
            coco_report.update(
                {
                    "status": "ok",
                    "samples": len(ds),
                    "first_id": sample.id if sample else None,
                    "data_path": str(data_path),
                    "labels_path": str(labels_path),
                }
            )
        except Exception as exc:  # pragma: no cover - env probe
            coco_report["status"] = "load_failed"
            coco_report["error"] = str(exc)
            report["status"] = "coco_check_failed"
            report["coco_validation"] = coco_report
            print(json.dumps(report, indent=2))
            sys.exit(1)
        finally:
            try:
                if "ds" in locals():
                    ds.delete()
            except Exception:
                pass

        report["coco_validation"] = coco_report

    report["status"] = "ok"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
