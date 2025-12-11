#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_blink_candidates.py

Merge BLINK candidate lists from two JSON files and write the merged result.
"""

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge BLINK entity candidate lists from two JSON files."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help=(
            "Path(s) to the primary BLINK JSON. "
            "If you supply two files separated by '::', ',', or ';', "
            "the first will be treated as file A and the second as file B."
        ),
    )
    parser.add_argument(
        "--second_input_file",
        default=None,
        help="Optional explicit path to the secondary BLINK JSON (file B).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the merged JSON will be written.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Filename for the merged JSON (e.g., merged.json).",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=10,
        help="Maximum number of candidates to keep per entity mention (default: 10).",
    )
    return parser.parse_args()


def split_dual_path(primary_arg: str, secondary_arg: str | None) -> Tuple[str, str]:
    if secondary_arg:
        return primary_arg, secondary_arg

    separators = ("::", ",", ";")
    for sep in separators:
        if sep in primary_arg:
            parts = [part.strip() for part in primary_arg.split(sep) if part.strip()]
            if len(parts) == 2:
                return parts[0], parts[1]

    raise ValueError(
        "You must either provide --second_input_file or pass two file paths "
        "joined by '::', ',' or ';' via --input_file."
    )


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def candidate_key(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    return str(candidate)


def merge_candidate_lists(
    list_a: Sequence[Any],
    list_b: Sequence[Any],
    max_len: int,
) -> List[Any]:
    merged: List[Any] = []
    seen: set[str] = set()

    def add_candidate(candidate: Any):
        key = candidate_key(candidate)
        if key not in seen:
            merged.append(candidate)
            seen.add(key)

    for source in (list_a, list_b):
        for candidate in source:
            if len(merged) >= max_len:
                return merged
            add_candidate(candidate)

    return merged


def build_candidate_lookup(
    mentions: Sequence[str],
    candidate_lists: Sequence[Sequence[Any]],
) -> Dict[str, Deque[Sequence[Any]]]:
    lookup: Dict[str, Deque[Sequence[Any]]] = defaultdict(deque)
    for mention, candidates in zip(mentions, candidate_lists):
        normalized = list(candidates) if isinstance(candidates, list) else []
        lookup[mention].append(normalized)
    return lookup


def ensure_list_length(lst: List[List[Any]], target: int):
    while len(lst) < target:
        lst.append([])
    return lst


def merge_blink_entity_candidates_list(
    file_a: str,
    file_b: str,
    output_path: Path,
    max_candidates: int,
):
    data_a = load_json(file_a)
    data_b = load_json(file_b)

    merged_data: Dict[str, Any] = {}
    total_docs = len(data_a)

    for idx, (doc_name, instance_a) in enumerate(data_a.items(), start=1):
        instance_b = data_b.get(doc_name)

        if "entities" not in instance_a:
            print(f"[WARN] Document {doc_name} missing 'entities' in file A; skipping.")
            continue

        entities_a = instance_a["entities"]
        entities_b = instance_b.get("entities") if instance_b else None

        mentions_a: List[str] = entities_a.get("entity_mentions", [])
        candidates_a: List[List[Any]] = entities_a.get(
            "blink_entity_candidates_list", []
        ) or []

        ensure_list_length(candidates_a, len(mentions_a))

        if not entities_b:
            print(f"[INFO] Doc {doc_name}: not found in file B; keeping file A candidates.")
            merged_candidates = [
                merge_candidate_lists(cand_a, [], max_candidates)
                for cand_a in candidates_a
            ]
        else:
            mentions_b: List[str] = entities_b.get("entity_mentions", [])
            candidates_b: List[List[Any]] = entities_b.get(
                "blink_entity_candidates_list", []
            ) or []

            lookup_b = build_candidate_lookup(mentions_b, candidates_b)
            merged_candidates: List[List[Any]] = []

            for mention, cand_a in zip(mentions_a, candidates_a):
                cand_b = lookup_b[mention].popleft() if lookup_b.get(mention) else []
                merged_candidates.append(
                    merge_candidate_lists(cand_a, cand_b, max_candidates)
                )

        entities_a["blink_entity_candidates_list"] = merged_candidates
        merged_data[doc_name] = instance_a

        print(f"[{idx}/{total_docs}] Processed document: {doc_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(merged_data, f_out, ensure_ascii=False, indent=4)

    print(f"[DONE] Final merged data saved to {output_path}")


def main():
    args = parse_args()
    file_a, file_b = split_dual_path(args.input_file, args.second_input_file)
    output_path = Path(args.output_dir).expanduser().resolve() / args.output_file

    print(f"[INFO] File A: {file_a}")
    print(f"[INFO] File B: {file_b}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Max candidates per mention: {args.max_candidates}")

    merge_blink_entity_candidates_list(
        file_a=file_a,
        file_b=file_b,
        output_path=output_path,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()