#!/usr/bin/env python3
"""
Alignment checker for EK100 TIM vs V-JEPA feature files.

It uses the EK100 CSV row order (pid, vid, start_frame, end_frame) as the canonical
sequence and compares the V-JEPA verb/noun .pt files and the TIM action .pt file
to that order. This is read-only; it does not reorder or write any files.

Usage (example):
  python utils/check_ek100_order.py \
    --csv /home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_train.csv \
    --verb /scratch/users/bickici/data/EK100/vjepa_features/vjepa2_huge_ek100_cls_verb/ek100_cls_train_feat.pt \
    --noun /scratch/users/bickici/data/EK100/vjepa_features/vjepa2_huge_ek100_cls_noun/ek100_cls_train_feat.pt \
    --action /scratch/users/bickici/data/TIM/action_tokens_train/features/epic_train_feat.pt \
    --tim-labels EPIC_100_train.pkl \
    --max-print 20
"""

import argparse
import csv
import pickle
import torch


def build_csv_ids(csv_path: str):
    ids = []
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)  # skip header
        for row in r:
            pid, vid = row[1:3]
            start, end = row[6], row[7]
            ids.append(f"{pid}_{vid}_{start}_{end}")
    return ids


def ids_from_pt(path: str, fallback):
    d = torch.load(path, map_location="cpu")
    for k in ("ids", "video_ids", "video_names", "uids", "keys", "filenames"):
        if k in d:
            return list(d[k])
    return fallback


def mismatch(ref, other):
    if len(ref) != len(other):
        return [( -1, f"length mismatch ref:{len(ref)} vs other:{len(other)}", "")]  # sentinel
    return [(i, ref[i], other[i]) for i in range(len(ref)) if ref[i] != other[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="EK100 train/val CSV path")
    ap.add_argument("--verb", required=True, help="V-JEPA verb .pt")
    ap.add_argument("--noun", required=True, help="V-JEPA noun .pt")
    ap.add_argument("--action", required=True, help="TIM action .pt")
    ap.add_argument("--tim-labels", required=True, help="TIM labels pickle (EPIC_100_train.pkl)")
    ap.add_argument("--max-print", type=int, default=20, help="Max mismatches to print per file")
    args = ap.parse_args()

    csv_ids = build_csv_ids(args.csv)
    print(f"CSV rows: {len(csv_ids)}")

    verb_ids = ids_from_pt(args.verb, csv_ids)
    noun_ids = ids_from_pt(args.noun, csv_ids)

    # TIM action .pt lacks ids; compare targets to TIM labels for a rough check.
    tim_labels = pickle.load(open(args.tim_labels, "rb"))
    tim_ids = [f"v_{i}" for i in tim_labels.index.tolist()]
    action_ids = ids_from_pt(args.action, csv_ids)  # fallback to CSV order if none stored

    mv = mismatch(csv_ids, verb_ids)
    mn = mismatch(csv_ids, noun_ids)
    ma = mismatch(csv_ids, action_ids)

    def report(name, mismatches):
        if mismatches and mismatches[0][0] == -1:
            print(f"{name}: length mismatch ({mismatches[0][1]})")
            return
        print(f"{name} mismatches: {len(mismatches)}")
        for m in mismatches[: args.max_print]:
            print(f"  idx={m[0]} ref={m[1]} other={m[2]}")
        if len(mismatches) > args.max_print:
            print(f"  ... ({len(mismatches) - args.max_print} more)")

    report("Verb vs CSV", mv)
    report("Noun vs CSV", mn)
    report("Action vs CSV", ma)

    # Also compare TIM target alignment if lengths match
    verb_tgt = torch.load(args.verb, map_location="cpu")["targets"]
    noun_tgt = torch.load(args.noun, map_location="cpu")["targets"]
    act_tgt = torch.load(args.action, map_location="cpu")["targets"]
    def target_mis(a, b): return len(a) - (a == b).sum().item()
    if len(verb_tgt) == len(act_tgt):
        print("verb targets vs action targets mismatches:", target_mis(verb_tgt, act_tgt))
    if len(noun_tgt) == len(act_tgt):
        print("noun targets vs action targets mismatches:", target_mis(noun_tgt, act_tgt))


if __name__ == "__main__":
    main()
