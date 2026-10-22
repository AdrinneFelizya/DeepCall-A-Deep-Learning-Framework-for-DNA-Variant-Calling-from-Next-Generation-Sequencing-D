"""
Evaluation Script for DNA Variant Calling.

Supports two modes:
  * synthetic: evaluate CNN/ResNet18/VGG16 on synthetic test pileups
    (from DL.zip) — accuracy, per-class precision/recall/F1, confusion matrix.
  * real: evaluate a trained model against a high-confidence truth VCF
    (e.g., GIAB NA12878 chr22) — site-level precision/recall/F1,
    genotype concordance, Ti/Tv ratio, and predicted VCF output.

Usage:
    python evaluate.py --mode synthetic --config config.yaml
    python evaluate.py --mode real --model checkpoints/best.pt \
        --config config.yaml --truth data/truth/NA12878.vcf \
        --out predictions.vcf
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import yaml

from config import *  # noqa: F401  (synthetic-mode constants)
from data.pileup_generator import PileupGenerator
from data.dataset import SyntheticPileupDataset
from models.cnn import VariantCNN
from models.resnet18 import VariantResNet18
from models.vgg16 import VariantVGG16
from models.transformer_caller import VariantCallerTransformer
from training.trainer import Trainer
from training.metrics import compute_metrics, plot_confusion_matrix
from postprocessing.vcf_writer import VCFWriter

# Real-data imports (optional)
try:
    import pysam
    from data.real_pileup import pileup_tensor, pileup_to_model_input
    from data.vcf_utils import (
        parse_truth_vcf, parse_pred_vcf,
        site_level_metrics, genotype_concordance, ti_tv_ratio,
    )
    _HAS_PYSAM = True
except Exception:
    _HAS_PYSAM = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device, num_classes: int,
                in_channels_override: int | None = None):
    m = config["model"]
    arch = m["architecture"]
    in_ch = in_channels_override if in_channels_override is not None else m["in_channels"]

    if arch == "cnn":
        model = VariantCNN(
            in_channels=in_ch, num_classes=num_classes,
            dropout_rate=m.get("dropout", 0.5),
        )
    elif arch == "resnet18":
        model = VariantResNet18(
            in_channels=in_ch, num_classes=num_classes,
            pretrained=m.get("pretrained", False),
            dropout_rate=m.get("dropout", 0.5),
        )
    elif arch == "vgg16":
        model = VariantVGG16(
            in_channels=in_ch, num_classes=num_classes,
            pretrained=m.get("pretrained", False),
            dropout_rate=m.get("dropout", 0.5),
        )
    elif arch == "transformer":
        t = m["transformer"]
        model = VariantCallerTransformer(
            in_channels=in_ch, num_classes=num_classes,
            d_model=t["d_model"], nhead=t["nhead"],
            num_layers=t["num_layers"],
            dim_feedforward=t["dim_feedforward"],
            dropout=t["dropout"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model.to(device)


def evaluate_synthetic(config: dict, device: torch.device):
    print("\n" + "=" * 60)
    print("DNA Variant Calling - Evaluation (SYNTHETIC mode)")
    print("=" * 60)

    d = config["data"]
    generator = PileupGenerator(
        height=d["synth_height"], width=d["synth_width"],
        channels=d["synth_channels"],
    )
    test_images, test_labels = generator.generate_dataset(
        d["num_test_samples"], seed=789
    )
    test_loader = torch.utils.data.DataLoader(
        SyntheticPileupDataset(test_images, test_labels),
        batch_size=config["train"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    d = config["data"]
    num_classes = config["model"].get("num_classes_synthetic", 4)
    class_names = ["No Variant", "SNP", "Insertion", "Deletion"]
    arch = config["model"]["architecture"]

    model = build_model(config, device, num_classes,
                        in_channels_override=d["synth_channels"])
    model_path = os.path.join(
        config["train"]["checkpoint_dir"], f"{arch}_best.pth"
    )

    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    trainer = Trainer(model=model, device=device, model_name=arch)
    predictions, labels, probs = trainer.predict(test_loader)
    metrics = compute_metrics(labels, predictions, num_classes=num_classes)

    print(f"\n--- {arch} Test Results ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for i, name in enumerate(class_names):
        print(
            f"  {name:12s} - P: {metrics['precision'][i]:.4f} | "
            f"R: {metrics['recall'][i]:.4f} | F1: {metrics['f1'][i]:.4f}"
        )

    os.makedirs(config.get("output_dir", "output"), exist_ok=True)
    cm_path = os.path.join(config["output_dir"], f"{arch}_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path=cm_path)

    vcf_path = os.path.join(config["output_dir"], f"{arch}_variants.vcf")
    vcf_writer = VCFWriter(vcf_path)
    positions = np.arange(len(predictions)) * 100 + 1000
    vcf_writer.write(predictions, probs, positions=positions)

    print(f"\nEvaluation complete! Results in: {config.get('output_dir', 'output')}")


def evaluate_real(
    config: dict, device: torch.device,
    model_path: str, truth_vcf: str, out_vcf: str,
    candidates_bed: str | None = None,
):
    print("\n" + "=" * 60)
    print("DNA Variant Calling - Evaluation (REAL mode)")
    print("=" * 60)

    if not _HAS_PYSAM:
        raise ImportError("pysam is required for real-data evaluation.")

    num_classes = config["model"]["num_classes"]  # 3
    model = build_model(config, device, num_classes)

    if not os.path.exists(model_path):
        print(f"Error: model checkpoint not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Truth set
    truth = parse_truth_vcf(truth_vcf)

    bam = pysam.AlignmentFile(config["data"]["bam_file"], "rb")
    threshold = config["evaluate"]["threshold"]

    # Candidate sites
    if candidates_bed and os.path.exists(candidates_bed):
        sites = []
        with open(candidates_bed) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                chrom = parts[0]
                pos = int(parts[1]) + 1
                ref = parts[4] if len(parts) > 4 else "."
                alt = parts[5] if len(parts) > 5 else "."
                sites.append((chrom, pos, ref, alt))
    else:
        sites = [(k[0], k[1], v.ref, v.alt) for k, v in truth.items()]

    # Inference
    pred_records = {}
    pred_list = []
    n_calls = 0

    print(f"Running inference on {len(sites)} candidate sites...")
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

        # Variant call: not hom-ref and confident
        is_call = (pred_label != 0) and (pred_prob >= threshold)
        if is_call:
            n_calls += 1
            out_alt = alt
            out_gt = {1: "0/1", 2: "1/1"}[pred_label]
            pred_records[(chrom, pos)] = __import__(
                "data.vcf_utils", fromlist=["VCFRecord"]
            ).VCFRecord(
                chrom=chrom, pos=pos, ref=ref, alt=out_alt,
                genotype=out_gt, qual=pred_prob,
            )
            pred_list.append((chrom, pos, ref, out_alt, out_gt, pred_prob))
        else:
            pred_list.append((chrom, pos, ref, ".", "0/0", 0.0))

    # Write VCF
    os.makedirs(os.path.dirname(out_vcf) or ".", exist_ok=True)
    with open(out_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##fileDate={__import__('datetime').datetime.now().strftime('%Y%m%d')}\n")
        f.write("##source=DeepCall-DL\n")
        f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
        f.write("##INFO=<ID=TYPE,Number=1,Type=String,Description=\"Variant Type\">\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for chrom, pos, ref, alt, gt, qual in pred_list:
            if alt == ".":
                continue
            f.write(
                f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual:.4f}\tPASS\t"
                f"DP=50;TYPE=VAR\tGT:GQ\t{gt}:{int(qual*100)}\n"
            )

    print(f"Wrote {n_calls} variant calls to {out_vcf}")

    # Metrics vs truth
    metrics = site_level_metrics(pred_records, truth, match_alleles=True)
    print("\nSite-level variant calling metrics:")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"  Precision={metrics['precision']:.4f}")
    print(f"  Recall={metrics['recall']:.4f}")
    print(f"  F1={metrics['f1']:.4f}")

    geno = genotype_concordance(pred_records, truth)
    print("\nGenotype concordance:")
    print(f"  {geno['genotype_matches']}/{geno['sites_compared']} "
          f"= {geno['genotype_concordance']:.4f}")

    ttv = ti_tv_ratio(pred_records)
    print(f"\nTi/Tv ratio (predicted SNPs): {ttv:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "real"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (real mode required)")
    parser.add_argument("--truth", type=str, default=None,
                        help="Truth VCF path (real mode)")
    parser.add_argument("--out", type=str, default="predictions.vcf",
                        help="Output VCF path")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Candidate BED (real mode)")
    args = parser.parse_args()

    config = load_config(args.config)

    if config["train"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["train"]["device"])

    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")

    if args.mode == "synthetic":
        evaluate_synthetic(config, device)
    else:
        if not args.model or not args.truth:
            parser.error("--model and --truth are required for real mode")
        evaluate_real(
            config, device,
            model_path=args.model,
            truth_vcf=args.truth,
            out_vcf=args.out,
            candidates_bed=args.candidates,
        )


if __name__ == "__main__":
    main()