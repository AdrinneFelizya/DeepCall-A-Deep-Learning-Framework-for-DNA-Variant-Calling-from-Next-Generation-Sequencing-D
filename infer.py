"""
Real inference: produce a VCF from a trained model on a new BAM.

Usage:
    python infer.py --model checkpoints/best.pt --config config.yaml \
        --bam data/bams/sample.bam --candidates data/processed/candidates.bed \
        --out sample_calls.vcf
"""

from __future__ import annotations

import argparse
import os
import torch
import yaml

try:
    import pysam
    from data.real_pileup import pileup_tensor, pileup_to_model_input
    _HAS_PYSAM = True
except Exception:
    _HAS_PYSAM = False

from evaluate import load_config, build_model


def main():
    if not _HAS_PYSAM:
        raise ImportError(
            "pysam is required for inference. "
            "Install via: conda install -c conda-forge -c bioconda pysam"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--bam", type=str, required=True)
    parser.add_argument("--candidates", type=str, required=True,
                        help="BED file of candidate sites to score")
    parser.add_argument("--out", type=str, default="calls.vcf")
    args = parser.parse_args()

    config = load_config(args.config)

    if config["train"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["train"]["device"])

    num_classes = config["model"]["num_classes"]
    model = build_model(config, device, num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    bam = pysam.AlignmentFile(args.bam, "rb")
    threshold = config["evaluate"]["threshold"]

    sites = []
    with open(args.candidates) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            pos = int(parts[1]) + 1
            ref = parts[4] if len(parts) > 4 else "."
            alt = parts[5] if len(parts) > 5 else "."
            sites.append((chrom, pos, ref, alt))

    print(f"Scoring {len(sites)} candidate sites...")
    predictions = []
    n_calls = 0

    for chrom, pos, ref, alt in sites:
        tensor = pileup_tensor(
            bam, chrom, pos,
            window=config["data"]["window_size"],
            max_reads=config["data"]["max_reads"],
            channels=config["data"]["channels"],
        )
        tensor = pileup_to_model_input(tensor)
        x = torch.from_numpy(tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_label = int(probs.argmax())
        pred_prob = float(probs[pred_label])
        is_call = (pred_label != 0) and (pred_prob >= threshold)

        if is_call:
            n_calls += 1
            out_alt = alt
            out_gt = {1: "0/1", 2: "1/1"}[pred_label]
            out_qual = pred_prob
        else:
            out_alt = "."
            out_gt = "0/0"
            out_qual = 0.0

        predictions.append((chrom, pos, ref, out_alt, out_gt, out_qual))

    # Write VCF
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(
            f"##fileDate={__import__('datetime').datetime.now().strftime('%Y%m%d')}\n"
        )
        f.write("##source=DeepCall-DL\n")
        f.write(
            "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n"
        )
        f.write(
            "##INFO=<ID=TYPE,Number=1,Type=String,Description=\"Variant Type\">\n"
        )
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write(
            "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n"
        )
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for chrom, pos, ref, alt, gt, qual in predictions:
            if alt == ".":
                continue
            f.write(
                f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual:.4f}\tPASS\t"
                f"DP=50;TYPE=VAR\tGT:GQ\t{gt}:{int(qual*100)}\n"
            )

    print(f"Done. {n_calls} variant calls written to {args.out}")


if __name__ == "__main__":
    main()