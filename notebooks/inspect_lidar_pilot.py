import json
from collections import Counter
from pathlib import Path

import laspy
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = PROJECT_ROOT / "data" / "raw" / "486000_5455000"


def inspect_file(path):
    class_counts = Counter()
    ground_count = 0
    ground_min_z = float("inf")
    ground_max_z = float("-inf")

    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()

        summary = {
            "file": str(path),
            "file_size_mb": round(path.stat().st_size / 1_000_000, 2),
            "las_version": str(header.version),
            "point_format": int(header.point_format.id),
            "point_count": int(header.point_count),
            "crs": crs.to_string() if crs else None,
            "crs_wkt": crs.to_wkt() if crs else None,
            "xyz_min": header.mins.tolist(),
            "xyz_max": header.maxs.tolist(),
            "xyz_scales": header.scales.tolist(),
            "xyz_offsets": header.offsets.tolist(),
        }

        for points in reader.chunk_iterator(1_000_000):
            classification = np.asarray(points.classification)

            classes, counts = np.unique(
                classification,
                return_counts=True,
            )

            class_counts.update({
                int(class_id): int(count)
                for class_id, count in zip(classes, counts)
            })

            # LAS ground classification.
            ground = classification == 2
            ground_count += int(ground.sum())

            if ground.any():
                ground_z = np.asarray(points.z)[ground]

                ground_min_z = min(
                    ground_min_z,
                    float(ground_z.min()),
                )
                ground_max_z = max(
                    ground_max_z,
                    float(ground_z.max()),
                )

    summary["classification_counts"] = dict(sorted(class_counts.items()))
    summary["ground_points"] = ground_count

    summary["ground_percent"] = round(
        100 * ground_count / summary["point_count"], 2
    ) if summary["point_count"] else 0

    summary["ground_elevation_range"] = {
        "min": ground_min_z if ground_count else None,
        "max": ground_max_z if ground_count else None,
    }

    return summary


def main():
    files = sorted(
        path for path in PILOT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".las", ".laz"}
    )

    if not files:
        raise FileNotFoundError(
            f"No extracted LAS/LAZ files found under {PILOT_DIR}"
        )

    for path in files:
        print(json.dumps(inspect_file(path), indent=2))


if __name__ == "__main__":
    main()