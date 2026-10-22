"""Real-time variant calling backend for DeepCall.

Provides ``RealTimeVariantCaller`` and ``write_vcf`` used by the Streamlit app.
Works with or without pysam: without pysam the model can still be loaded, but
BAM scanning requires pysam.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    import pysam
    _HAS_PYSAM = True
except Exception:  # pragma: no cover
    pysam = None
    _HAS_PYSAM = False

try:
    from data.real_pileup import pileup_tensor, pileup_to_model_input, pysam as _pysam
    _HAS_REAL_PILEUP = _pysam is not None
except Exception:  # pragma: no cover
    pileup_tensor = None
    pileup_to_model_input = None
    _HAS_REAL_PILEUP = False

CLASS_NAMES_REAL = ["Hom-Ref", "Het", "Hom-Alt"]
CLASS_NAMES_SYNTH = ["No Variant", "SNP", "Insertion", "Deletion"]


def _infer_architecture(model_path: str) -> str:
    name = os.path.basename(model_path).lower()
    for arch in ("resnet18", "transformer", "vgg16", "cnn"):
        if arch in name:
            return arch
    return "cnn"


def _state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _infer_channels(arch: str, state: dict, fallback: int) -> int:
    key_map = {
        "cnn": "conv1.0.weight",
        "resnet18": "resnet.conv1.weight",
        "vgg16": "vgg.features.0.weight",
        "transformer": "input_proj.weight",
    }
    key = key_map.get(arch)
    if key in state:
        w = state[key]
        if w.dim() == 4:
            return int(w.shape[1])
        if w.dim() == 2:
            return int(w.shape[1])
    # Fallback: search
    for k, v in state.items():
        if k.endswith(".weight") and v.dim() in (2, 4):
            # Heuristic: first conv/linear with small-ish in-features
            if v.dim() == 4 and v.shape[1] in (3, 5, 8, 1, 4):
                return int(v.shape[1])
    return fallback


def _infer_num_classes(arch: str, state: dict, fallback: int) -> int:
    # Find last classifier weight
    cands = [k for k in state.keys() if "classifier" in k and k.endswith(".weight")]
    if cands:
        k = sorted(cands)[-1]
        return int(state[k].shape[0])
    # Transformer / others
    for k in ("classifier.weight", "resnet.fc.1.weight"):
        if k in state:
            return int(state[k].shape[0])
    return fallback


class RealTimeVariantCaller:
    """Loads a trained model and runs variant calling on BAM regions."""

    def __init__(self, config_path: str = "config.yaml", model_path: str = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = None
        self.architecture = None
        self.in_channels = None
        self.num_classes = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str, architecture: str = None):
        if architecture is None:
            architecture = _infer_architecture(model_path)
        self.architecture = architecture
        self.model_name = architecture

        state = _state_dict(model_path)

        # Channel / class counts
        cfg_channels = self.config["data"].get("channels", 8)
        cfg_classes = self.config["model"].get("num_classes", 3)
        in_ch = _infer_channels(architecture, state, cfg_channels)
        n_cls = _infer_num_classes(architecture, state, cfg_classes)

        self.in_channels = in_ch
        self.num_classes = n_cls

        # Class names
        if n_cls == 3:
            self.class_names = CLASS_NAMES_REAL
        else:
            self.class_names = CLASS_NAMES_SYNTH[:n_cls]

        d = self.config["data"]
        if architecture == "cnn":
            from models.cnn import VariantCNN

            self.model = VariantCNN(in_channels=in_ch, num_classes=n_cls)
        elif architecture == "resnet18":
            from models.resnet18 import VariantResNet18

            self.model = VariantResNet18(
                in_channels=in_ch, num_classes=n_cls, pretrained=False
            )
        elif architecture == "vgg16":
            from models.vgg16 import VariantVGG16

            self.model = VariantVGG16(in_channels=in_ch, num_classes=n_cls, pretrained=False)
        elif architecture == "transformer":
            from models.transformer_caller import VariantCallerTransformer

            t = self.config["model"].get("transformer", {})
            self.model = VariantCallerTransformer(
                in_channels=in_ch,
                num_classes=n_cls,
                d_model=t.get("d_model", 128),
                nhead=t.get("nhead", 8),
                num_layers=t.get("num_layers", 4),
                dim_feedforward=t.get("dim_feedforward", 512),
                dropout=t.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {architecture} model from {model_path}")

    def _adapt_tensor(self, tensor: np.ndarray) -> np.ndarray:
        # tensor: (channels, reads, window)
        if tensor.shape[0] == self.in_channels:
            return tensor.astype(np.float32)
        out = np.zeros((self.in_channels, tensor.shape[1], tensor.shape[2]), dtype=np.float32)
        n = min(tensor.shape[0], self.in_channels)
        out[:n] = tensor[:n]
        if tensor.shape[0] < self.in_channels:
            # pad remaining channels with zeros (already zero)
            pass
        return out

    def predict_site(self, bam, chrom: str, pos: int, ref: str = "A", alt: str = "G"):
        if not _HAS_PYSAM or not _HAS_REAL_PILEUP:
            raise ImportError("pysam is required for BAM scanning.")
        cfg = self.config["data"]
        tensor = pileup_tensor(
            bam,
            chrom,
            pos,
            window=cfg.get("window_size", 251),
            max_reads=cfg.get("max_reads", 250),
            channels=cfg.get("channels", 8),
        )
        tensor = pileup_to_model_input(tensor)
        tensor = self._adapt_tensor(tensor)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])
        threshold = self.config.get("evaluate", {}).get("threshold", 0.9)
        is_variant = (pred_idx != 0) and (conf >= threshold)

        # Map real classes: 0 hom-ref, 1 het, 2 hom-alt
        if self.num_classes == 3:
            class_name = self.class_names[pred_idx]
            gt = {0: "0/0", 1: "0/1", 2: "1/1"}.get(pred_idx, "0/0")
            out_alt = alt if is_variant else "."
        else:
            # Synthetic-style model: treat non-zero as variant
            class_name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
            gt = "0/1" if is_variant else "0/0"
            out_alt = alt if is_variant else "."

        vcf_line = ""
        if is_variant:
            vcf_line = (
                f"{chrom}\t{pos}\t.\t{ref}\t{out_alt}\t{conf:.4f}\tPASS\t"
                f"DP=50;TYPE=VAR\tGT:GQ\t{gt}:{int(conf*100)}"
            )

        return {
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": out_alt if is_variant else alt,
            "class": pred_idx,
            "class_name": class_name,
            "confidence": conf,
            "is_variant": is_variant,
            "vcf_line": vcf_line,
        }

    def batch_predict_from_positions(self, bam_path: str, sites: List[tuple]) -> List[Dict[str, Any]]:
        if not _HAS_PYSAM:
            raise ImportError("pysam is required for BAM scanning.")
        bam = pysam.AlignmentFile(bam_path, "rb")
        results = []
        for site in sites:
            chrom = site[0]
            pos = int(site[1])
            ref = site[2] if len(site) > 2 else "A"
            alt = site[3] if len(site) > 3 else "G"
            r = self.predict_site(bam, chrom, pos, ref, alt)
            if r["is_variant"]:
                results.append(r)
        try:
            bam.close()
        except Exception:
            pass
        return results


def write_vcf(variants: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        import datetime as _dt

        f.write(f"##fileDate={_dt.datetime.now().strftime('%Y%m%d')}\n")
        f.write("##source=DeepCall-DL\n")
        f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">\n')
        f.write('##INFO=<ID=TYPE,Number=1,Type=String,Description="Variant Type">\n')
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for v in variants:
            if v.get("vcf_line"):
                f.write(v["vcf_line"] + "\n")
            elif v.get("is_variant"):
                f.write(
                    f"{v['chrom']}\t{v['pos']}\t.\t{v['ref']}\t{v['alt']}\t"
                    f"{v['confidence']:.4f}\tPASS\tDP=50;TYPE=VAR\t"
                    f"GT:GQ\t0/1:{int(v['confidence']*100)}\n"
                )
    print(f"Wrote {len(variants)} variant calls to {out_path}")


__all__ = ["RealTimeVariantCaller", "write_vcf"]