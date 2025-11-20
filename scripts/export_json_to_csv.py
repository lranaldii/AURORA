import json
import csv
import argparse


def jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(
        csv_path, "w", encoding="utf-8", newline=""
    ) as f_out:
        writer = None
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if writer is None:
                fields = list(record.keys())
                writer = csv.DictWriter(f_out, fieldnames=fields)
                writer.writeheader()
            writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export JSONL scenarios to CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default="aurora/data/aurora_scenarios.jsonl",
        help="Path to JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aurora/data/aurora_scenarios.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()
    jsonl_to_csv(args.input, args.output)
    print(f"Exported {args.input} to {args.output}")
