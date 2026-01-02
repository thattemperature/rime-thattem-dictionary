#!/usr/bin/env python3
"""
Rime Dictionary Weight Distribution Analyzer
Analyzes and visualizes the distribution of weights in a Rime dictionary file.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_rime_dict(filepath):
    """
    Parse a Rime dictionary file and extract weights.

    Args:
        filepath: Path to the .dict.yaml file

    Returns:
        tuple: (weights_list, entries_without_weight, total_entries)
    """
    weights = []
    entries_without_weight = 0
    total_entries = 0
    in_data_section = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Check if we've passed the header (after "...")
            if line == '...':
                in_data_section = True
                continue

            # Skip header section
            if not in_data_section:
                continue

            # Parse data lines: word<tab>code<tab>weight
            # or: word<tab>code (no weight)
            parts = line.split('\t')

            if len(parts) >= 2:
                total_entries += 1

                # Check if there's a weight (last part should be numeric)
                if len(parts) >= 3:
                    # Weight might be after the code, could have multiple codes separated by spaces
                    weight_str = parts[-1].strip()
                    try:
                        weight = int(weight_str)
                        weights.append(weight)
                    except ValueError:
                        # Not a valid weight, count as no weight
                        entries_without_weight += 1
                else:
                    entries_without_weight += 1

    return weights, entries_without_weight, total_entries


def analyze_distribution(weights):
    """
    Analyze the distribution of weights.

    Args:
        weights: List of weight values

    Returns:
        dict: Statistics about the weight distribution
    """
    if not weights:
        return None

    counter = Counter(weights)

    stats = {
        'total_with_weight': len(weights),
        'unique_weights': len(counter),
        'min': min(weights),
        'max': max(weights),
        'mean': sum(weights) / len(weights),
        'median': sorted(weights)[len(weights) // 2],
        'most_common': counter.most_common(10)
    }

    return stats


def create_histogram(weights, bins=10):
    """
    Create a text-based histogram of weight distribution.

    Args:
        weights: List of weight values
        bins: Number of bins for the histogram

    Returns:
        str: Text representation of histogram
    """
    if not weights:
        return "No weights to display"

    min_w = min(weights)
    max_w = max(weights)
    bin_width = (max_w - min_w) / bins

    # Create bins
    bin_counts = defaultdict(int)
    for w in weights:
        if bin_width == 0:
            bin_idx = 0
        else:
            bin_idx = min(int((w - min_w) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1

    # Find max count for scaling
    max_count = max(bin_counts.values()) if bin_counts else 1
    scale = 50 / max_count  # Scale to 50 characters width

    # Build histogram
    result = []
    result.append("\nWeight Distribution Histogram:")
    result.append("=" * 60)

    for i in range(bins):
        bin_start = min_w + i * bin_width
        bin_end = bin_start + bin_width
        count = bin_counts.get(i, 0)
        bar = '█' * int(count * scale)
        result.append(f"{bin_start:6.0f}-{bin_end:6.0f}: {bar} ({count})")

    return '\n'.join(result)


def print_report(filepath, weights, entries_without_weight, total_entries):
    """
    Print a comprehensive report of the weight distribution.
    """
    print(f"\n{'='*60}")
    print(f"Rime Dictionary Weight Analysis Report")
    print(f"{'='*60}")
    print(f"File: {filepath}")
    print(f"\nTotal entries: {total_entries}")
    print(f"Entries with weight: {len(weights)}")
    print(f"Entries without weight: {entries_without_weight}")
    print(f"Coverage: {len(weights)/total_entries*100:.1f}%")

    if not weights:
        print("\nNo weights found in the dictionary file.")
        return

    stats = analyze_distribution(weights)

    print(f"\n{'='*60}")
    print("Weight Statistics:")
    print(f"{'='*60}")
    print(f"Unique weight values: {stats['unique_weights']}")
    print(f"Minimum weight: {stats['min']}")
    print(f"Maximum weight: {stats['max']}")
    print(f"Mean weight: {stats['mean']:.2f}")
    print(f"Median weight: {stats['median']}")

    print(f"\n{'='*60}")
    print("Top 10 Most Common Weights:")
    print(f"{'='*60}")
    print(f"{'Weight':<10} {'Count':<10} {'Percentage'}")
    print(f"{'-'*60}")
    for weight, count in stats['most_common']:
        percentage = count / len(weights) * 100
        print(f"{weight:<10} {count:<10} {percentage:>5.2f}%")

    print(create_histogram(weights))

    # Weight ranges analysis
    print(f"\n{'='*60}")
    print("Weight Range Distribution:")
    print(f"{'='*60}")
    ranges = [
        (0, 0, "Zero weight"),
        (1, 10, "Low (1-10)"),
        (11, 50, "Medium (11-50)"),
        (51, 99, "High (51-99)"),
        (100, float('inf'), "Very High (100+)")
    ]

    for low, high, label in ranges:
        count = sum(1 for w in weights if low <= w <= high)
        percentage = count / len(weights) * 100
        print(f"{label:<20} {count:>6} entries ({percentage:>5.2f}%)")


def main():
    if len(sys.argv) != 2:
        print("Usage: python rime_weight_analyzer.py <dict_file.dict.yaml>")
        print("\nExample: python rime_weight_analyzer.py rime_ice.base.dict.yaml")
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    if not filepath.suffix == '.yaml' and not str(filepath).endswith('.dict.yaml'):
        print("Warning: File doesn't have .dict.yaml extension. Proceeding anyway...")

    try:
        weights, entries_without_weight, total_entries = parse_rime_dict(
            filepath)
        print_report(filepath, weights, entries_without_weight, total_entries)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
