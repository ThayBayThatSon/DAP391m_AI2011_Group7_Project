from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diagnostics import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_PREDICTION_PATH,
    sync_prediction_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize historical model predictions into SQLite."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    count = sync_prediction_csv(args.predictions, args.database)
    print(f"Synchronized {count:,} prediction rows into {args.database}")


if __name__ == "__main__":
    main()
