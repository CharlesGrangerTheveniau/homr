"""
Count dynamics markings and hairpins in the Lieder MusicXML dataset.

Scans all MusicXML files and reports:
- Total files with dynamics
- Total dynamic events by type
- Crescendo/diminuendo hairpin counts
"""

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
lieder = os.path.join(dataset_root, "Lieder-main")


def count_dynamics_in_file(file_path: str) -> tuple[Counter[str], Counter[str]]:
    """Parse a MusicXML file and count dynamics and wedge elements."""
    dynamics_counter: Counter[str] = Counter()
    wedge_counter: Counter[str] = Counter()

    try:
        tree = ET.parse(file_path)  # noqa: S314
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  XML parse error in {file_path}: {e}", file=sys.stderr)
        return dynamics_counter, wedge_counter

    for direction in root.iter("direction"):
        for direction_type in direction.iter("direction-type"):
            for dynamics in direction_type.iter("dynamics"):
                for child in dynamics:
                    tag = child.tag
                    if tag and tag not in ("other-dynamics",):
                        dynamics_counter[tag] += 1

            for wedge in direction_type.iter("wedge"):
                wedge_type = wedge.get("type", "unknown")
                wedge_counter[wedge_type] += 1

    return dynamics_counter, wedge_counter


def main() -> None:
    search_dirs = [
        os.path.join(lieder, "flat"),
        os.path.join(lieder, "scores"),
    ]

    musicxml_files: list[str] = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for path in Path(search_dir).rglob("*.musicxml"):
                musicxml_files.append(str(path))

    if not musicxml_files:
        print(f"No MusicXML files found under {lieder}")
        print("Run the Lieder dataset conversion first (convert_lieder.py)")
        sys.exit(1)

    print(f"Scanning {len(musicxml_files)} MusicXML files...\n")

    total_dynamics: Counter[str] = Counter()
    total_wedges: Counter[str] = Counter()
    files_with_dynamics = 0
    files_with_wedges = 0
    files_processed = 0

    for file_path in sorted(musicxml_files):
        dynamics, wedges = count_dynamics_in_file(file_path)
        files_processed += 1

        if sum(dynamics.values()) > 0:
            files_with_dynamics += 1
        if sum(wedges.values()) > 0:
            files_with_wedges += 1

        total_dynamics += dynamics
        total_wedges += wedges

        if files_processed % 100 == 0:
            print(f"  Processed {files_processed}/{len(musicxml_files)} files...", file=sys.stderr)

    print("=" * 60)
    print("DYNAMICS IN LIEDER DATASET")
    print("=" * 60)
    print(f"\nFiles scanned:          {len(musicxml_files)}")
    print(f"Files with dynamics:    {files_with_dynamics}")
    print(f"Files with hairpins:    {files_with_wedges}")
    print(f"\nTotal dynamic markings: {sum(total_dynamics.values())}")
    print(f"Total wedge elements:   {sum(total_wedges.values())}")

    print("\n--- Dynamic Markings by Type ---")
    for marking, count in total_dynamics.most_common():
        print(f"  {marking:20s} {count:>8d}")

    print("\n--- Wedge (Hairpin) Elements by Type ---")
    for wedge_type, count in total_wedges.most_common():
        print(f"  {wedge_type:20s} {count:>8d}")

    # Summary of tokens we'd use
    token_dynamics = {"ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "sfz", "fp"}
    covered = sum(count for mark, count in total_dynamics.items() if mark in token_dynamics)
    uncovered = sum(count for mark, count in total_dynamics.items() if mark not in token_dynamics)

    print(f"\n--- Coverage Analysis ---")
    print(f"  Dynamics covered by vocabulary:   {covered:>8d}")
    print(f"  Dynamics NOT in vocabulary:        {uncovered:>8d}")
    if covered + uncovered > 0:
        print(f"  Coverage ratio:                   {covered / (covered + uncovered) * 100:.1f}%")

    hairpin_starts = total_wedges.get("crescendo", 0) + total_wedges.get("diminuendo", 0)
    hairpin_stops = total_wedges.get("stop", 0)
    print(f"\n  Hairpin starts:  {hairpin_starts:>8d}")
    print(f"  Hairpin stops:   {hairpin_stops:>8d}")
    if hairpin_starts > 0:
        print(f"  Stop/Start ratio: {hairpin_stops / hairpin_starts:.2f}")


if __name__ == "__main__":
    main()
