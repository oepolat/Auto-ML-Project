from __future__ import annotations

from pathlib import Path
from automl.data import Dataset
from automl.automl import AutoML
from automl.utils import set_seeds
import argparse

import logging

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()

DATADIR = FILE.parent / "data"

def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path
):
    set_seeds(seed)

    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("Loading AutoML")

    automl = AutoML(task_id=task, output_path=output_path, seed=seed, dataset=dataset)
    automl.load_and_test_best_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["y_prop", "bike_sharing", "brazilian_houses", "exam_dataset"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/exam_dataset/1/predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on."
            " You are free to also evaluate on other folds for your own analysis."
            " For the test dataset we will only provide a single fold, fold 1."
        ),
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help=(
            "The directory where the datasets are stored."
            " You should be able to mostly leave this as the default."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running task {args.task}"
        f"\n{args}"
    )

    main(
        task=args.task,
        fold=args.fold,
        output_path=args.output_path,
        datadir=args.datadir,
        seed=args.seed
    )
