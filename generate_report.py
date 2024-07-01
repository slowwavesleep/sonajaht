import argparse
import json
from pathlib import Path
import uuid

from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from metrics import calculate_all_metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file_path", type=str)
    group.add_argument("--files_dir", type=str)

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 25, 50, 100],
        help="One or more `k` value",
    )
    parser.add_argument("--small", action="store_true")
    return parser.parse_args()


def evaluate_file(
    name: str | Path,
    k_list: list[int],
    count_self: bool,
    count_synonyms: bool,
    small: bool,
) -> dict[str, int | float | str]:
    logger.info(f"Reading {name}")
    predictions = []
    with open(name) as file:
        for line in file:
            predictions.append(json.loads(line))
    logger.info(f"Total number of predictions: {len(predictions)}")
    metrics = calculate_all_metrics(
        predictions,
        k_list,
        count_self=count_self,
        count_synonyms=count_synonyms,
        small=small,
    )
    return metrics


def main():
    args = parse_arguments()
    reports_path = Path("./reports").resolve()
    reports_path.mkdir(exist_ok=True)
    report_base_name = str(uuid.uuid4())
    size = "small" if args.small else "large"
    if args.files_dir:
        files_path = Path(args.files_dir).resolve()
        files = [file for file in files_path.glob("*.jsonl") if file.is_file()]
        only_syns_list = []
        self_and_syns_list = []
        only_self_list = []
        for file in tqdm(files):
            only_syns_list.append(
                dict(name=file.stem)
                | evaluate_file(
                    file,
                    k_list=args.k,
                    count_self=False,
                    count_synonyms=True,
                    small=args.small,
                )
            )
            self_and_syns_list.append(
                dict(name=file.stem)
                | evaluate_file(
                    file,
                    k_list=args.k,
                    count_self=True,
                    count_synonyms=True,
                    small=args.small,
                )
            )
            only_self_list.append(
                dict(name=file.stem)
                | evaluate_file(
                    file,
                    k_list=args.k,
                    count_self=True,
                    count_synonyms=False,
                    small=args.small,
                )
            )

        only_syns_report_file = reports_path / f"{report_base_name}-syns-{size}.md"
        self_and_syns_report_file = reports_path / f"{report_base_name}-all-{size}.md"
        only_self_report_file = reports_path / f"{report_base_name}-self-{size}.md"
        with open(only_syns_report_file, "w") as file:
            file.write(
                tabulate(only_syns_list, headers="keys", tablefmt="github") + "\n"
            )
        with open(self_and_syns_report_file, "w") as file:
            file.write(
                tabulate(self_and_syns_list, headers="keys", tablefmt="github") + "\n"
            )
        with open(only_self_report_file, "w") as file:
            file.write(
                tabulate(only_self_list, headers="keys", tablefmt="github") + "\n"
            )
    elif args.file_path:
        file = Path(args.file_path).resolve()
        only_syns = evaluate_file(
            file,
            k_list=args.k,
            count_self=False,
            count_synonyms=True,
            small=args.small,
        )
        self_and_syns = evaluate_file(
            file,
            k_list=args.k,
            count_self=True,
            count_synonyms=True,
            small=args.small,
        )
        only_self = evaluate_file(
            file,
            k_list=args.k,
            count_self=True,
            count_synonyms=False,
            small=args.small,
        )

        keys = only_syns.keys()
        transposed_data = [
            [key] + [record[key] for record in [only_syns, self_and_syns, only_self]]
            for key in keys
        ]
        headers = ["", "syns", "all", "self"]
        single_file_report = reports_path / f"{str(uuid.uuid4())}-{file.stem}-{size}.md"
        with open(single_file_report, "w") as file:
            file.write(
                tabulate(transposed_data, tablefmt="github", headers=headers) + "\n"
            )
    else:
        raise ValueError


if __name__ == "__main__":
    main()
