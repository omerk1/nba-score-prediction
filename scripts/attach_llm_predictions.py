"""
Attach the LLM's raw p_play to a context frame as the `llm_p_play` column.

Calibrating or stacking the LLM needs its prediction on TRAINING rows too, not
just evaluated ones, so this runs the estimator over every row of the frame.
Anything already in the response cache costs nothing; only the earliest season
(never evaluated, therefore never called) triggers new calls on a first run.

Usage:
  venv/bin/python3 scripts/attach_llm_predictions.py --context-cache <in.parquet> --out <out.parquet>
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.availability.llm_derived import LLM_COLUMN
from src.availability.llm_estimator import LLMEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="anonymized", choices=["anonymized", "named"])
    args = ap.parse_args()

    ctx = pd.read_parquet(args.context_cache)
    est = LLMEstimator(variant=args.variant)
    ctx[LLM_COLUMN] = est.predict(ctx)
    logger.info(f"{est.n_failed} rows without a prediction of {len(ctx)}")
    ctx.to_parquet(args.out)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
