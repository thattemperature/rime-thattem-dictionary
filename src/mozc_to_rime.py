#!/usr/bin/env python3
"""
Mozc to Rime Dictionary Converter
Converts Google Mozc dictionary format to Rime dictionary format.

Mozc format (TSV):
  reading<tab>lid<tab>rid<tab>cost<tab>word

  Example:
  あいかわらず	839	219	5084	あいかわらず
  あいかわらず	839	219	5393	相変わらず

Rime format (YAML + TSV):
  ---
  name: dict_name
  version: "LTS"
  sort: by_weight
  ...
  word<tab>reading<tab>weight
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
from collections import defaultdict


class MozcToRimeConverter:
    def __init__(self):
        """
        Initialize the converter.
        """
        self.entries = []
        self.min_cost = float("inf")
        self.max_cost = float("-inf")

    def parse_mozc_line(
        self, line: str
    ) -> Optional[Tuple[str, str, int, int, int, str]]:
        """
        Parse a line from Mozc dictionary.

        Args:
            line: A line from Mozc dictionary file

        Returns:
            Tuple of (reading, lid, rid, cost, word) or None if invalid
        """
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            return None

        parts = line.split("\t")

        # Mozc format: reading<tab>lid<tab>rid<tab>cost<tab>word
        if len(parts) != 5:
            return None

        try:
            reading = parts[0].strip()
            lid = int(parts[1].strip())
            rid = int(parts[2].strip())
            cost = int(parts[3].strip())
            word = parts[4].strip()

            # Skip if reading or word is empty
            if not reading or not word:
                return None

            # Track min/max cost for normalization
            self.min_cost = min(self.min_cost, cost)
            self.max_cost = max(self.max_cost, cost)

            return (reading, lid, rid, cost, word)

        except (ValueError, IndexError):
            return None

    def cost_to_weight(self, cost: int) -> float:
        """
        Convert Mozc cost to Rime weight.
        In Mozc: lower cost = higher priority (costs are likely -log(probability) based)
        In Rime: higher weight = higher priority

        Since Mozc costs are logarithmic, we use exponential decay for conversion.

        Args:
            cost: Mozc cost value

        Returns:
            Rime weight value (float, not limited to specific range)
        """
        import math

        # Mozc costs typically range from ~1000 to ~15000
        # Lower cost = more common = higher weight

        # Using exponential decay: weight = max_weight * exp(-k * cost)
        # Reference point: cost 3000 maps to high weight
        base_cost = 8192
        decay_constant = 2 / base_cost

        weight = 128 * math.exp(-decay_constant * (cost - base_cost))

        return weight

    def load_mozc_dict(self, filepath: Path) -> int:
        """
        Load Mozc dictionary file.
        Merges entries with same word+reading combination.

        Args:
            filepath: Path to Mozc dictionary file

        Returns:
            Number of entries loaded
        """

        # Dictionary to accumulate weights for same word+reading
        # Key: (word, reading), Value: list of weights
        entries_dict = defaultdict(list)
        temp_entries = []
        count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = self.parse_mozc_line(line)
                    if result:
                        temp_entries.append(result)
                        count += 1
                except Exception as e:
                    print(
                        f"Warning: Error parsing line {line_num}: {e}", file=sys.stderr
                    )
                    continue

        # Convert costs to weights and group by word+reading
        for reading, lid, rid, cost, word in temp_entries:
            weight = self.cost_to_weight(cost)
            entries_dict[(word, reading)].append(weight)

        # Merge entries: for same word+reading, combine weights
        # Since weight ~ exp(-cost) and cost ~ -log(P), we have weight ~ P
        # So we should sum the weights (combining probabilities)
        for (word, reading), weights in entries_dict.items():
            # Sum weights to combine probabilities from different contexts
            combined_weight = sum(weights)
            # Store the min cost for reference (best/lowest cost)
            self.entries.append((word, reading, combined_weight, min(weights)))

        return count

    def generate_rime_dict(
        self,
        output_path: Path,
        dict_name: str,
        version: str = "LTS",
        sort: str = "by_weight",
        include_cost_comment: bool = False,
    ):
        """
        Generate Rime dictionary file.

        Args:
            output_path: Output file path
            dict_name: Dictionary name
            version: Dictionary version
            sort: Sort method ('by_weight' or 'original')
            include_cost_comment: Include original Mozc cost as comment
        """
        with open(output_path, "w", encoding="utf-8") as f:
            # Write YAML header
            f.write("# Rime dictionary\n")
            f.write("# encoding: utf-8\n")
            f.write("# Converted from Mozc dictionary\n")
            f.write("# Conversion: exponential decay from Mozc cost to Rime weight\n")
            f.write("---\n")
            f.write(f"name: {dict_name}\n")
            f.write(f'version: "{version}"\n')
            f.write(f"sort: {sort}\n")

            if include_cost_comment:
                f.write("columns:\n")
                f.write("  - text\n")
                f.write("  - code\n")
                f.write("  - weight\n")
                f.write("  - stem  # Original Mozc cost\n")

            f.write("...\n")

            # Write entries
            for word, reading, weight, _ in self.entries:
                # Convert weight to integer for output
                int_weight = int(round(weight))
                if include_cost_comment:
                    f.write(f"{word}\t{reading}\t{int_weight}\t# merged\n")
                else:
                    f.write(f"{word}\t{reading}\t{int_weight}\n")

    def get_statistics(self) -> dict:
        """
        Get statistics about the loaded dictionary.

        Returns:
            Dictionary with statistics
        """
        if not self.entries:
            return {}

        weights = [w for _, _, w, _ in self.entries]

        return {
            "total_entries": len(self.entries),
            "unique_words": len(set(w for w, _, _, _ in self.entries)),
            "unique_readings": len(set(r for _, r, _, _ in self.entries)),
            "min_weight": min(weights),
            "max_weight": max(weights),
            "avg_weight": sum(weights) / len(weights),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Google Mozc dictionary to Rime dictionary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dictionary00.txt -o rime_dict.dict.yaml
  %(prog)s dictionary00.txt -o rime_dict.dict.yaml -n my_dict
  %(prog)s dictionary00.txt -o rime_dict.dict.yaml --stats --include-cost

Mozc dictionary format (TSV):
  reading<tab>lid<tab>rid<tab>cost<tab>word

  Example:
  あいかわらず	839	219	5084	あいかわらず
  あいかわらず	839	219	5393	相変わらず

Rime dictionary format:
  ---
  name: dict_name
  version: "LTS"
  sort: by_weight
  ...
  word<tab>reading<tab>weight

Conversion method:
  Uses exponential decay to convert Mozc cost (lower=better) to Rime weight (higher=better).
  This respects the logarithmic nature of language model costs.
  Entries with same word+reading are merged by summing their weights.
        """,
    )

    parser.add_argument(
        "input", type=str, help="Input Mozc dictionary file (TSV format)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output Rime dictionary file (.dict.yaml)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Dictionary name (default: derived from filename)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="LTS",
        help="Dictionary version (default: LTS)",
    )
    parser.add_argument(
        "-s",
        "--sort",
        type=str,
        default="by_weight",
        choices=["by_weight", "original"],
        help="Sort method (default: by_weight)",
    )
    parser.add_argument(
        "--include-cost",
        action="store_true",
        help="Include original Mozc cost as comment in output",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show statistics after conversion"
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    # Determine dictionary name
    dict_name = args.name
    if not dict_name:
        # Use filename without extension
        dict_name = output_path.stem
        if dict_name.endswith(".dict"):
            dict_name = dict_name[:-5]

    # Convert
    print(f"Converting Mozc dictionary: {input_path}")
    print(f"Output Rime dictionary: {output_path}")
    print(f"Dictionary name: {dict_name}")
    print()

    converter = MozcToRimeConverter()

    try:
        # Load Mozc dictionary
        count = converter.load_mozc_dict(input_path)
        print(f"Loaded {count} raw entries from Mozc dictionary")
        print(f"Merged into {len(converter.entries)} unique word+reading combinations")

        if len(converter.entries) == 0:
            print("Warning: No valid entries found in input file", file=sys.stderr)
            sys.exit(1)

        # Generate Rime dictionary
        converter.generate_rime_dict(
            output_path,
            dict_name,
            version=args.version,
            sort=args.sort,
            include_cost_comment=args.include_cost,
        )
        print(f"Successfully created Rime dictionary: {output_path}")

        # Show statistics
        if args.stats:
            stats = converter.get_statistics()
            print("\n" + "=" * 60)
            print("Conversion Statistics:")
            print("=" * 60)
            print(f"Total entries: {stats['total_entries']}")
            print(f"Unique words: {stats['unique_words']}")
            print(f"Unique readings: {stats['unique_readings']}")
            print(
                f"Weight range: {stats['min_weight']:.2f} - {stats['max_weight']:.2f}"
            )
            print(f"Average weight: {stats['avg_weight']:.2f}")

        print("\nConversion completed successfully!")
        print("Remember to deploy your Rime input method to use the new dictionary.")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
