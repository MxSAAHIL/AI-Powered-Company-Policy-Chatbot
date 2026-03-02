from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATASET_PATH
from app.data_loader import inspect_dataset


def main() -> None:
    result = inspect_dataset(str(DATASET_PATH))
    print(f"Dataset shape: {result['shape']}")
    print("Columns:")
    for column in result["columns"]:
        print(f"  - {column}")

    print(f"\nDetected knowledge column: {result['detected_column']}\n")
    print("Column scores:")
    for score in result["scores"]:
        print(
            f"  - {score['name']}: score={score['score']:.3f}, "
            f"avg_length={score['avg_length']:.1f}, "
            f"multi_word_ratio={score['multi_word_ratio']:.2f}"
        )

    print("\nSample rows:")
    pprint(result["sample_rows"])


if __name__ == "__main__":
    main()
