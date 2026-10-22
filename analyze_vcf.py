"""Analyze a VCF file and print human-readable mutation summaries.

Usage:
    python analyze_vcf.py --vcf data/truth/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf --limit 20
"""
from __future__ import annotations

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from data.vcf_utils import parse_truth_vcf


def classify(ref: str, alt: str) -> str:
    if alt == ".":
        return "No variant"
    if len(ref) == 1 and len(alt) == 1:
        return "SNP (single base changed)"
    if len(alt) > len(ref):
        return "Insertion (extra bases added)"
    if len(ref) > len(alt):
        return "Deletion (bases removed)"
    return "Complex variant"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", type=str, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    truth = parse_truth_vcf(args.vcf)

    counts = {}
    examples = []
    for (chrom, pos), rec in list(truth.items())[: args.limit * 5]:
        label = classify(rec.ref, rec.alt)
        counts[label] = counts.get(label, 0) + 1
        if len(examples) < args.limit:
            examples.append((chrom, pos, rec.ref, rec.alt, label, rec.genotype))

    print(f"Total sites parsed: {len(truth)}")
    print("\nVariant type counts (first scan):")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print(f"\nFirst {len(examples)} examples:")
    for chrom, pos, ref, alt, label, gt in examples:
        print(f"  {chrom}:{pos}  {ref} -> {alt}  | {label} | genotype={gt}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("chrom\tpos\tref\talt\ttype\tgenotype\n")
            for (chrom, pos), rec in truth.items():
                f.write(f"{chrom}\t{pos}\t{rec.ref}\t{rec.alt}\t{classify(rec.ref, rec.alt)}\t{rec.genotype}\n")
        print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    main()
```<|end|><|start|>write to<|channel|>assistant<|channel|>analysis<|message|>Created analyze_vcf.py. Run it on downloaded VCF.<|end|><|start|>start to=shell<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
cd "C:\Users\adrin\OneDrive\Desktop\DL project\deepcall_real"; python analyze_vcf.py --vcf data/truth/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf --limit 10