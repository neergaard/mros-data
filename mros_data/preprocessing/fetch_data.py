import argparse
import sys
from typing import Optional

from mros_data.preprocessing.fetch_functions import download_fns
from mros_data.utils.logger import get_logger

logger = get_logger()

AVAILABLE_COHORTS = set(download_fns.keys())


def download_cohort(output_dir: str, cohort: str, n_first: Optional[int] = None):

    download_fns[cohort](output_dir, n_first)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="Path to output directory.\nWill be created if not available."
    )
    parser.add_argument("-c", "--cohort", type=str, choices=AVAILABLE_COHORTS, help="Available cohorts.", required=True)
    parser.add_argument("-n", "--n_first", type=int, help="Number of recordings to download.")
    args = parser.parse_args()

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------------------------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == (len(vars(args)) - 1):
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    download_cohort(args.output_dir, args.cohort, args.n_first)
