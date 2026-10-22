# DeepCall-Real — DNA Variant Calling from Real NGS Data

Enhanced deep learning pipeline for identifying genetic mutations from next-generation sequencing (NGS) data.

**Built on your `DL.zip` starter** (CNN / ResNet18 / VGG16 + synthetic pileups) and extended for **real-life application** and **conference publication**.

## What this project does

| Step | Description |
|------|-------------|
| 1 | Align FASTQ → BAM (BWA-MEM) / use existing BAM |
| 2 | Encode real read pileups around candidate sites as multi-channel images |
| 3 | Train DNN (CNN, ResNet18, VGG16, or Transformer) to classify genotype: **hom-ref / het / hom-alt** |
| 4 | Evaluate against a high-confidence truth set (e.g., GIAB NA12878) — precision, recall, F1, Ti/Tv |
| 5 | Output standard **VCF** ready for clinical annotation (VEP, ClinVar) |
| 6 | Grad-CAM interpretability — visualize which reads drive each call |

## Real-life applications

- **Clinical diagnostics** — germline variants for rare-disease diagnosis
- **Precision oncology** — somatic mutations (extend to tumor/normal pairs)
- **Pharmacogenomics** — drug-response variant screening
- **Infectious-disease surveillance** — pathogen mutation tracking

## Project structure

```
deepcall_real/
├── README.md
├── config.yaml             # Hyperparameters & paths
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── pileup_generator.py # (from DL.zip) synthetic pileups for smoke test
│   ├── real_pileup.py      # NEW: BAM/CRAM → real pileup tensor (pysam)
│   ├── dataset.py          # (from DL.zip) PyTorch Dataset
│   ├── vcf_utils.py        # NEW: truth VCF parsing, site-level metrics
│   └── make_dataset.py     # NEW: candidate-site BED from truth VCF
├── models/
│   ├── __init__.py
│   ├── cnn.py              # (from DL.zip) VariantCNN
│   ├── resnet18.py         # (from DL.zip) VariantResNet18
│   ├── vgg16.py            # (from DL.zip) VariantVGG16
│   ├── transformer_caller.py  # NEW: attention-based novelty model
│   └── grad_cam.py         # NEW: interpretability heatmaps
├── training/
│   ├── __init__.py
│   ├── trainer.py          # (from DL.zip) Trainer with early stopping
│   ├── metrics.py          # (from DL.zip) + site-level variant metrics
│   └── focal_loss.py       # NEW: focal loss for extreme class imbalance
├── postprocessing/
│   ├── __init__.py
│   └── vcf_writer.py       # (from DL.zip) VCF generation
├── train.py                # (from DL.zip) enhanced: synthetic OR real mode
├── evaluate.py             # (from DL.zip) enhanced: truth-set comparison
├── infer.py                # NEW: real inference → VCF from a BAM
├── smoke_test.py           # (from DL.zip) quick synthetic run
├── paper/
│   ├── outline.md          # conference paper outline
│   └── template.tex        # LaTeX paper template (ISMB / RECOMB / BIBM style)
└── notebooks/
    └── 01_eda.py
```

## Quick start

### 1. Install dependencies

```bash
cd "C:\Users\DELL\OneDrive\Desktop\DL project\deepcall_real"
pip install -r requirements.txt
```

> **Note on `pysam`:** On Windows, `pysam` often fails to compile from source. If `pip install pysam` errors, use conda:
> ```bash
> conda install -c conda-forge -c bioconda pysam
> ```
> The **smoke test and synthetic training work without pysam**. Real BAM processing needs it.

### 2. Smoke test (synthetic data, no pysam needed)

```bash
python smoke_test.py
```
Trains CNN/ResNet18/VGG16 on synthetic pileups and writes VCFs in minutes.

### 3. Real-data pipeline (needs pysam + a BAM)

Download public benchmark data (no IRB required):
- **GIAB NA12878** high-confidence truth VCF + BAM (NIST Genome in a Bottle)
- **GRCh38 reference FASTA**

Place them under:
```
data/refs/GRCh38.fa
data/bams/NA12878.bam (+ .bai)
data/truth/NA12878.vcf
```

Then:
```bash
# Generate candidate-site BED from truth VCF
python -m data.make_dataset --truth data/truth/NA12878.vcf --out data/processed

# Train on real pileups (chromosome-wise split, no data leakage)
python train.py --mode real --config config.yaml

# Evaluate against truth (precision/recall/F1, Ti/Tv, genotype concordance)
python evaluate.py --mode real --model checkpoints/best.pt --config config.yaml \
    --truth data/truth/NA12878.vcf --out predictions.vcf

# Interpretability: Grad-CAM heatmap for a specific site
python -m models.grad_cam --model checkpoints/best.pt --bam data/bams/NA12878.bam \
    --chr chr22 --pos 17345678 --out gradcam_site.png
```

### 4. Inference on a new sample

```bash
python infer.py --model checkpoints/best.pt --config config.yaml \
    --bam data/bams/sample.bam --candidates data/processed/candidates.bed \
    --out sample_calls.vcf
```

## Conference paper roadmap

1. **Baseline** — replicate your DL.zip results on synthetic data (CNN vs ResNet18 vs VGG16)
2. **Real data** — train/evaluate on GIAB chr22, compare to GATK HaplotypeCaller / DeepVariant
3. **Add novelty** — Transformer encoder, Grad-CAM interpretability, cross-population generalization
4. **Write paper** — use `paper/template.tex` + `paper/outline.md`
5. **Target venues** — ISMB/ECCB, RECOMB, IEEE BIBM, PSB, NeurIPS/ICLR workshops
6. **Release** — GitHub + Zenodo DOI for reproducibility

## Testing

```bash
# Import / model construction tests
python tests/test_imports.py

# Full synthetic smoke test (CNN, ResNet18, VGG16)
python smoke_test.py
```

## Troubleshooting
- `ModuleNotFoundError: streamlit/plotly`: `pip install -r requirements.txt`
- `pysam` on Windows: use conda or skip real-data mode (synthetic mode works without it).
- VGG16 training uses a reduced learning rate automatically for stability.

## Ethics

This is a **research prototype**, not a clinically validated diagnostic. Use only de-identified public data. Do not upload patient BAM/VCF files to public repositories without IRB approval and consent.