"""Script to prepare datasets."""

import argparse
import os

from tinytools import get_logger

from tinyhumans.datasets.prepare.prepare_behave import prepare as prepare_behave_dataset
from tinyhumans.datasets.prepare.prepare_grab import prepare as prepare_grab_dataset
from tinyhumans.datasets.prepare.prepare_interact import prepare as prepare_interact_dataset
from tinyhumans.datasets.prepare.prepare_intercap import prepare as prepare_intercap_dataset
from tinyhumans.datasets.prepare.prepare_omomo import prepare as prepare_omomo_dataset

logger = get_logger(__name__, "info")


PREPARATION_FUNCTIONS = {
    "behave": prepare_behave_dataset,
    "grab": prepare_grab_dataset,
    "interact": prepare_interact_dataset,
    "intercap": prepare_intercap_dataset,
    "omomo": prepare_omomo_dataset,
}


def main() -> None:
    """Dispatcher to prepare a specified dataset."""
    parser = argparse.ArgumentParser(
        description="Download and prepare various motion capture datasets.",
        formatter_class=argparse.RawTextHelpFormatter,  # Allows for better help text formatting
    )

    # A positional argument to specify which dataset to prepare
    parser.add_argument(
        "dataset_name",
        choices=PREPARATION_FUNCTIONS.keys(),
        help="The name of the dataset to prepare.\nAvailable choices: " + ", ".join(PREPARATION_FUNCTIONS.keys()),
    )
    parser.add_argument("-o", "--output", required=True, help="The destination folder for the dataset.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress download progress bar and informational messages."
    )
    parser.add_argument("--username", help="Username for GRAB and InterCap datasets.")
    parser.add_argument("--password", help="Password for GRAB and InterCap datasets.")

    args = parser.parse_args()

    # --- Get Credentials (prioritize args, then env vars) ---
    username = (
        args.username or os.environ.get(f"{args.dataset_name.upper()}_USERNAME") or os.environ.get("MPI_USERNAME")
    )
    password = (
        args.password or os.environ.get(f"{args.dataset_name.upper()}_PASSWORD") or os.environ.get("MPI_PASSWORD")
    )

    # Get the correct preparation function from our dictionary
    prepare_function = PREPARATION_FUNCTIONS.get(args.dataset_name)

    logger.info("Starting preparation for the '%s' dataset...", args.dataset_name)
    # Call the selected function with the parsed arguments
    kwargs = {"output_dir": args.output, "quiet": args.quiet}
    if args.dataset_name in {"intercap", "grab"}:
        if not all([username, password]):
            msg = "Username and password are required for InterCap and GRAB datasets."
            raise ValueError(msg)
        kwargs |= {"username": username, "password": password}

    prepare_function(**kwargs)

    if not args.quiet:
        logger.info("Successfully prepared the '%s' dataset.", args.dataset_name)


if __name__ == "__main__":
    main()
