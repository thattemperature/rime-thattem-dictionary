#!/usr/bin/env python3
"""
BNC to Rime Dictionary Converter
Converts BNC word frequency list to Rime dictionary format.

BNC format (space-separated):
  frequency word pos file_count

  Example:
  25781 a_few dt0 3145
  1234 hello uj0 567

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
import re
from pathlib import Path
from typing import Tuple, Optional, Callable
from collections import defaultdict


class BNCToRimeConverter:
    # Pattern to extract [a-z]+ core from words
    EXTRACT_PATTERN = re.compile(r"[a-z]+")

    def __init__(self, code_function: Optional[Callable[[str], str]] = None):
        """
        Initialize the converter.

        Args:
            code_function: Function to convert word to Rime input code
                          If None, a placeholder function will be used
        """
        self.entries = []
        self.code_function = code_function or self._default_code_function
        self.filtered_count = 0  # Track number of filtered words

    def _normalize_word(self, word: str) -> Optional[str]:
        """
        Normalize and validate word according to rules:
        1. Extract [a-z]+ part if surrounded by other characters
        2. -_' should not be at the beginning
        3. -_' should not appear consecutively (two or more in a row)

        Args:
            word: The word to normalize

        Returns:
            Normalized word if valid, None if should be filtered out
        """
        if not all(c.isalpha() or c in "-_'" for c in word):
            match = self.EXTRACT_PATTERN.search(word)
            return match.group(0) if match else None

        if word and word[0] in "-_'":
            match = self.EXTRACT_PATTERN.search(word)
            return match.group(0) if match else None

        special_chars = "-_'"
        for i in range(len(word) - 1):
            if word[i] in special_chars and word[i + 1] in special_chars:
                match = self.EXTRACT_PATTERN.search(word)
                return match.group(0) if match else None

        return word

    def _default_code_function(self, word: str) -> str:
        """
        Default placeholder function for converting word to input code.
        Currently returns the word itself as lowercase.
        Replace this with your actual conversion function.

        Args:
            word: The word to convert

        Returns:
            Input code for the word
        """
        # TODO: Replace with actual code conversion function
        # Examples of what this could be:
        # - Pinyin for Chinese characters
        # - Romanization for other scripts
        # - Phonetic representation
        # - Custom encoding scheme
        return word.lower()

    def parse_bnc_line(self, line: str) -> Optional[Tuple[int, str, str, int]]:
        """
        Parse a line from BNC frequency list.

        Args:
            line: A line from BNC file

        Returns:
            Tuple of (frequency, word, pos, file_count) or None if invalid
        """
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            return None

        parts = line.split()

        # BNC format: frequency word pos file_count
        if len(parts) != 4:
            return None

        try:
            frequency = int(parts[0].strip())
            word = parts[1].strip()
            pos = parts[2].strip()
            file_count = int(parts[3].strip())

            # Skip if word is empty
            if not word:
                return None

            return (frequency, word, pos, file_count)

        except (ValueError, IndexError):
            return None

    def load_bnc_list(self, filepath: Path) -> int:
        """
        Load BNC frequency list file.
        Merges entries with same word but different POS by summing frequencies.

        Args:
            filepath: Path to BNC frequency list file

        Returns:
            Number of raw entries loaded
        """
        # Dictionary to accumulate frequencies for same word
        # Key: word, Value: list of (frequency, pos) tuples
        word_freq_dict = defaultdict(list)
        count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = self.parse_bnc_line(line)
                    if result:
                        frequency, word, pos, file_count = result
                        # Normalize and validate word
                        normalized_word = self._normalize_word(word)
                        if normalized_word is not None:
                            word_freq_dict[normalized_word].append((frequency, pos))
                            count += 1
                except Exception as e:
                    print(
                        f"Warning: Error parsing line {line_num}: {e}", file=sys.stderr
                    )
                    continue

        # Merge entries: sum frequencies for same word
        for normalized_word, freq_pos_list in word_freq_dict.items():

            # Sum all frequencies for this word
            total_frequency = sum(freq for freq, _ in freq_pos_list)

            # Get input code for this word
            try:
                code = self.code_function(normalized_word)
            except Exception as e:
                print(
                    f"Warning: Error generating code for word '{normalized_word}': {e}",
                    file=sys.stderr,
                )
                # Use default fallback
                code = self._default_code_function(normalized_word)

            # Store: (word, code, weight)
            # Using frequency directly as weight (higher frequency = higher weight)
            self.entries.append((normalized_word, code, total_frequency))

        self.entries.sort()
        return count

    def generate_rime_dict(
        self,
        output_path: Path,
        dict_name: str,
        version: str = "LTS",
        sort: str = "by_weight",
    ):
        """
        Generate Rime dictionary file.

        Args:
            output_path: Output file path
            dict_name: Dictionary name
            version: Dictionary version
            sort: Sort method ('by_weight' or 'original')
        """
        with open(output_path, "w", encoding="utf-8") as f:
            # Write YAML header
            f.write("# Rime dictionary\n")
            f.write("# encoding: utf-8\n")
            f.write("# Converted from BNC word frequency list\n")
            f.write("# Frequency is used directly as weight\n")
            f.write("---\n")
            f.write(f"name: {dict_name}\n")
            f.write(f'version: "{version}"\n')
            f.write(f"sort: {sort}\n")
            f.write("...\n")

            # Write entries
            # Rime format: word<tab>code<tab>weight
            for word, code, weight in self.entries:
                f.write(f" {word}\t{code}\t{weight}\n")

    def get_statistics(self) -> dict:
        """
        Get statistics about the loaded dictionary.

        Returns:
            Dictionary with statistics
        """
        if not self.entries:
            return {}

        weights = [w for _, _, w in self.entries]

        return {
            "total_entries": len(self.entries),
            "unique_words": len(set(w for w, _, _ in self.entries)),
            "unique_codes": len(set(c for _, c, _ in self.entries)),
            "min_weight": min(weights),
            "max_weight": max(weights),
            "avg_weight": sum(weights) / len(weights),
            "total_frequency": sum(weights),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert BNC word frequency list to Rime dictionary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s bnc_freq.txt -o rime_dict.dict.yaml
  %(prog)s bnc_freq.txt -o rime_dict.dict.yaml -n english_dict
  %(prog)s bnc_freq.txt -o rime_dict.dict.yaml --stats

BNC format (space-separated):
  frequency word pos file_count

  Example:
  25781 a_few dt0 3145
  1234 hello uj0 567

Rime dictionary format:
  ---
  name: dict_name
  version: "LTS"
  sort: by_weight
  ...
  word<tab>code<tab>weight

Conversion method:
  - Words with different POS tags are merged into one entry
  - Frequencies are summed for the same word
  - The frequency is used directly as weight in Rime
  - A code_function is used to convert words to input codes (customize as needed)
        """,
    )

    parser.add_argument("input", type=str, help="Input BNC frequency list file")
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
    print(f"Converting BNC frequency list: {input_path}")
    print(f"Output Rime dictionary: {output_path}")
    print(f"Dictionary name: {dict_name}")
    print()

    # TODO: Replace with your actual code conversion function
    # Example: converter = BNCToRimeConverter(code_function=my_pinyin_function)
    converter = BNCToRimeConverter()

    try:
        # Load BNC frequency list
        count = converter.load_bnc_list(input_path)
        print(f"Loaded {count} raw entries from BNC frequency list")
        print(f"Filtered out {converter.filtered_count} words with invalid characters")
        print(f"Merged into {len(converter.entries)} unique words")

        if len(converter.entries) == 0:
            print("Warning: No valid entries found in input file", file=sys.stderr)
            sys.exit(1)

        # Generate Rime dictionary
        converter.generate_rime_dict(
            output_path,
            dict_name,
            version=args.version,
            sort=args.sort,
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
            print(f"Unique codes: {stats['unique_codes']}")
            print(f"Weight range: {stats['min_weight']} - {stats['max_weight']}")
            print(f"Average weight: {stats['avg_weight']:.2f}")
            print(f"Total frequency: {stats['total_frequency']}")

        print("\nConversion completed successfully!")
        print("NOTE: Words are normalized according to these rules:")
        print("      - Extract [a-z]+ core from words with surrounding characters")
        print("      - Characters -_' cannot be at the beginning")
        print("      - Characters -_' cannot appear consecutively")
        print("      The default code function uses the normalized word as code.")
        print("      Modify the code_function parameter to use your custom conversion.")
        print("Remember to deploy your Rime input method to use the new dictionary.")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
