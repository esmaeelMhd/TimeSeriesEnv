"""Small command-line entry point for package smoke checks."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="python -m time_series_env",
        description="TimeSeriesEnv package entry point.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print a short package confirmation message.",
    )
    args = parser.parse_args()

    if args.version:
        print("TimeSeriesEnv package is importable.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
