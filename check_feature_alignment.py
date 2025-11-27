#!/usr/bin/env python3
"""
Lightweight checker to verify verb / noun / action feature files share the same ordering.
Usage (example):
  python utils/check_feature_alignment.py \
    --verb path/to/verb.pt \
    --noun path/to/noun.pt \
    --action path/to/action.pt \
    --canonical verb \
    --save-misaligned noun_misaligned.txt

If the order is mismatched, the script prints the first few mismatches and can
write all mismatching indices to a text file. It can also emit aligned copies
when you provide --save-<modality>-aligned.
"""

import argparse
from typing import Any, Dict, Iterable, List, Sequence
import torch

DEFAULT_ID_KEYS = (
    "ids",
    "video_ids",
    "video_names",
    "uids",
    "keys",
    "filenames",
)


def find_id_key(d: Dict[str, Any], user_key: str = None) -> str:
    """Pick an id-like key from a feature dict."""
    if user_key:
        if user_key not in d:
            raise KeyError(f"Requested id key '{user_key}' not found. Available keys: {list(d.keys())}")
        return user_key
    for k in DEFAULT_ID_KEYS:
        if k in d:
            return k
    return ""  # handled by caller


def reorder_container(container: Any, order: Sequence[int]) -> Any:
    """Reorder a tensor/list-like container by a permutation."""
    if hasattr(container, "__getitem__"):
        try:
            return container[order]
        except Exception:
            pass
    if isinstance(container, list):
        return [container[i] for i in order]
    raise TypeError(f"Don't know how to reorder type: {type(container)}")


def align_dict(data: Dict[str, Any], order: Sequence[int]) -> Dict[str, Any]:
    """Return a shallow copy of data with feat-like entries reordered."""
    out = {}
    for k, v in data.items():
        if isinstance(v, (torch.Tensor, list)):
            out[k] = reorder_container(v, order)
        else:
            out[k] = v
    return out


def load_id_list(path: str) -> List[Any]:
    """Load an external id list (either json array/dict values or newline-separated text)."""
    import json
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return list(data.values())
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unsupported JSON structure in {path}; expected list or dict.")
    # Fallback: text file with one id per line
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_ids(path: str, id_key: str = None, fallback_list: str = None) -> (Dict[str, Any], List[Any], str):
    data = torch.load(path, map_location="cpu")
    key = find_id_key(data, id_key)
    if key:
        ids = list(data[key])
        return data, ids, key
    if fallback_list:
        ids = load_id_list(fallback_list)
        return data, ids, f"external:{fallback_list}"
    raise KeyError(
        f"No id-like key found and no fallback list provided. Checked keys {DEFAULT_ID_KEYS}. "
        f"Available keys: {list(data.keys())}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check alignment of verb/noun/action feature files.")
    parser.add_argument("--verb", required=True, help="Path to verb feature .pt file")
    parser.add_argument("--noun", required=True, help="Path to noun feature .pt file")
    parser.add_argument("--action", required=True, help="Path to action/TIM feature .pt file")
    parser.add_argument("--verb-id-key", default=None, help="Override id key for verb file")
    parser.add_argument("--noun-id-key", default=None, help="Override id key for noun file")
    parser.add_argument("--action-id-key", default=None, help="Override id key for action file")
    parser.add_argument("--verb-list", default=None, help="External id list for verb file (json or text, one id per line)")
    parser.add_argument("--noun-list", default=None, help="External id list for noun file (json or text, one id per line)")
    parser.add_argument("--action-list", default=None, help="External id list for action file (json or text, one id per line)")
    parser.add_argument("--canonical-list", default=None,
                        help="External canonical id list (json or text). Used if canonical file lacks ids.")
    parser.add_argument("--canonical", choices=["verb", "noun", "action"], default="verb",
                        help="Which file defines the canonical ordering")
    parser.add_argument("--max-print", type=int, default=20, help="Max mismatches to print")
    parser.add_argument("--save-misaligned", type=str, default=None,
                        help="Path to write all mismatching indices (text file)")
    parser.add_argument("--save-verb-aligned", type=str, default=None,
                        help="Optional path to save aligned verb file")
    parser.add_argument("--save-noun-aligned", type=str, default=None,
                        help="Optional path to save aligned noun file")
    parser.add_argument("--save-action-aligned", type=str, default=None,
                        help="Optional path to save aligned action file")
    args = parser.parse_args()

    verb_data, verb_ids, verb_key = load_ids(args.verb, args.verb_id_key, args.verb_list)
    noun_data, noun_ids, noun_key = load_ids(args.noun, args.noun_id_key, args.noun_list)
    act_data, act_ids, act_key = load_ids(args.action, args.action_id_key, args.action_list)

    if not (len(verb_ids) == len(noun_ids) == len(act_ids)):
        raise ValueError(f"Length mismatch: verb={len(verb_ids)}, noun={len(noun_ids)}, action={len(act_ids)}")

    if args.canonical == "verb":
        canon = verb_ids
        canon_label = verb_key
    elif args.canonical == "noun":
        canon = noun_ids
        canon_label = noun_key
    else:
        canon = act_ids
        canon_label = act_key

    if not canon_label and args.canonical_list:
        canon = load_id_list(args.canonical_list)
        canon_label = f"external:{args.canonical_list}"
        if len(canon) != len(verb_ids):
            raise ValueError("Canonical list length does not match feature file length.")
    elif not canon_label:
        raise ValueError(
            f"Canonical file '{args.canonical}' has no ids. Provide --canonical-list to supply ordering."
        )

    def make_index(ids: Iterable[Any]) -> Dict[Any, int]:
        idx = {}
        for i, k in enumerate(ids):
            if k in idx:
                raise ValueError(f"Duplicate id '{k}' encountered; cannot build a unique index.")
            idx[k] = i
        return idx

    canon_index = make_index(canon)

    def compute_order(target_ids: List[Any], name: str):
        order = []
        mismatches = []
        for i, k in enumerate(canon):
            j = canon_index.get(k, None)
            if j is None:
                mismatches.append((i, k, None))
                order.append(None)
            else:
                if target_ids[i] != k:
                    mismatches.append((i, k, target_ids[i]))
                order.append(j)
        return order, mismatches

    noun_order, noun_mis = compute_order(noun_ids, "noun")
    act_order, act_mis = compute_order(act_ids, "action")

    total_mis = noun_mis + act_mis
    print(f"Verb id key: {verb_key}, Noun id key: {noun_key}, Action id key: {act_key}")
    print(f"Total samples: {len(canon)}")
    print(f"Noun mismatches: {len(noun_mis)}, Action mismatches: {len(act_mis)}")

    def print_mis(label: str, mis):
        if not mis:
            print(f"{label}: aligned")
            return
        print(f"{label}: first {min(len(mis), args.max_print)} mismatches (idx, canonical, found):")
        for m in mis[: args.max_print]:
            print(f"  {m}")
        if len(mis) > args.max_print:
            print(f"  ... ({len(mis) - args.max_print} more)")

    print_mis("Noun", noun_mis)
    print_mis("Action", act_mis)

    if args.save_misaligned and total_mis:
        with open(args.save_misaligned, "w") as f:
            for m in total_mis:
                f.write(f"{m}\n")
        print(f"Wrote mismatching indices to {args.save_misaligned}")

    # Save aligned copies if requested
    if args.save_verb_aligned:
        torch.save(verb_data, args.save_verb_aligned)
        print(f"Saved verb (canonical) copy to {args.save_verb_aligned}")
    if args.save_noun_aligned and len(noun_mis) > 0:
        aligned_noun = align_dict(noun_data, noun_order)
        torch.save(aligned_noun, args.save_noun_aligned)
        print(f"Saved aligned noun file to {args.save_noun_aligned}")
    if args.save_action_aligned and len(act_mis) > 0:
        aligned_act = align_dict(act_data, act_order)
        torch.save(aligned_act, args.save_action_aligned)
        print(f"Saved aligned action file to {args.save_action_aligned}")


if __name__ == "__main__":
    main()
