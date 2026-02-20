#!/usr/bin/env python3
"""
Parse metrics.txt files and update the results CSV template.

Usage:
    python parse_results.py <model_name> <dataset> <metrics_file>

Example:
    python parse_results.py EAGER Beauty results/eager_Beauty_20260215_230630/metrics.txt
"""

import re
import sys
import csv
from pathlib import Path


def parse_metrics_file(file_path):
    """Parse a metrics.txt file and extract all metrics."""
    metrics = {
        1: {'Precision': None, 'Recall': None, 'MRR': None, 'MAP': None, 'NDCG': None},
        5: {'Precision': None, 'Recall': None, 'MRR': None, 'MAP': None, 'NDCG': None},
        10: {'Precision': None, 'Recall': None, 'MRR': None, 'MAP': None, 'NDCG': None},
        20: {'Precision': None, 'Recall': None, 'MRR': None, 'MAP': None, 'NDCG': None},
        100: {'Precision': None, 'Recall': None, 'MRR': None, 'MAP': None, 'NDCG': None},
    }

    pattern = r'@(\d+):\s*Precision=([\d.]+),\s*Recall=([\d.]+),\s*MRR=([\d.]+),\s*MAP=([\d.]+),\s*NDCG=([\d.]+)'

    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                k = int(match.group(1))
                if k in metrics:
                    metrics[k]['Precision'] = float(match.group(2))
                    metrics[k]['Recall'] = float(match.group(3))
                    metrics[k]['MRR'] = float(match.group(4))
                    metrics[k]['MAP'] = float(match.group(5))
                    metrics[k]['NDCG'] = float(match.group(6))

    return metrics


def flatten_metrics(metrics):
    """Convert metrics dict to flat list in CSV order."""
    ks = [1, 5, 10, 20, 100]
    metric_names = ['Precision', 'Recall', 'MRR', 'MAP', 'NDCG']

    flat = []
    for k in ks:
        for name in metric_names:
            flat.append(metrics[k][name])
    return flat


def read_csv_template(csv_path):
    """Read existing CSV template and return data."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    return header, rows


def write_csv(csv_path, header, rows):
    """Write CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def find_and_update_row(rows, model, dataset, new_values):
    """Find matching row and update values. Returns True if found and updated."""
    for row in rows:
        if len(row) >= 2 and row[0] == model and row[1] == dataset:
            # Update from index 2 onwards
            for i, val in enumerate(new_values):
                if 2 + i < len(row):
                    row[2 + i] = val
            return True
    return False


def add_new_row(rows, model, dataset, new_values, num_cols):
    """Add a new row for model/dataset combination."""
    row = [model, dataset] + new_values
    # Pad to match number of columns
    while len(row) < num_cols:
        row.append('')
    rows.append(row)


def main():
    if len(sys.argv) < 4:
        print("Usage: python parse_results.py <model_name> <dataset> <metrics_file>")
        print("\nExample:")
        print("  python parse_results.py EAGER Beauty results/eager_Beauty_20260215_230630/metrics.txt")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset = sys.argv[2]
    metrics_file = sys.argv[3]

    # Parse metrics file
    print(f"Parsing {metrics_file}...")
    metrics = parse_metrics_file(metrics_file)
    flat_values = flatten_metrics(metrics)

    print(f"\nParsed metrics for {model_name} on {dataset}:")
    ks = [1, 5, 10, 20, 100]
    for k in ks:
        print(f"@{k}: P={metrics[k]['Precision']:.6f}, R={metrics[k]['Recall']:.6f}, "
              f"MRR={metrics[k]['MRR']:.6f}, MAP={metrics[k]['MAP']:.6f}, NDCG={metrics[k]['NDCG']:.6f}")

    # Get CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "results_template.csv"

    if not csv_path.exists():
        print(f"\nError: CSV template not found at {csv_path}")
        sys.exit(1)

    # Read and update CSV
    header, rows = read_csv_template(csv_path)
    num_cols = len(header)

    if find_and_update_row(rows, model_name, dataset, flat_values):
        print(f"\nUpdated existing row for {model_name} - {dataset}")
    else:
        add_new_row(rows, model_name, dataset, flat_values, num_cols)
        print(f"\nAdded new row for {model_name} - {dataset}")

    write_csv(csv_path, header, rows)
    print(f"Updated {csv_path}")


if __name__ == "__main__":
    main()
