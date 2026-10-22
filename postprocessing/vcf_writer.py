"""
VCF File Generation from Model Predictions.
(from DL.zip, enhanced) Converts model predictions into standard VCF v4.2.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime


class VCFWriter:
    """Converts model predictions to VCF format."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def write(
        self,
        predictions,
        probabilities,
        positions=None,
        chrom="chr1",
    ):
        """
        Write predictions to VCF file.

        Args:
            predictions: Array of predicted class labels (0-3 for synthetic,
                         or 0=hom-ref/1=het/2=hom-alt for real).
            probabilities: Array of prediction probabilities (softmax output).
            positions: Array of genomic positions (optional).
            chrom: Chromosome name.
        """
        with open(self.output_path, "w") as f:
            # Header
            f.write("##fileformat=VCFv4.2\n")
            f.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\n")
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
            f.write(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
            )

            # Write variants (skip class 0 - no variant / hom-ref)
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if pred == 0:  # Skip no-variant / hom-ref
                    continue

                pos = positions[i] if positions is not None else (i + 1) * 100
                confidence = prob[pred] * 100

                # Determine variant type and alleles
                if pred == 1:  # SNP / het
                    ref, alt = self._get_snp_alleles()
                    qual = int(confidence)
                    info_type = "SNP"
                    gt = "0/1"
                elif pred == 2:  # Insertion / hom-alt (synthetic: insertion)
                    ref, alt = self._get_insertion_alleles()
                    qual = int(confidence)
                    info_type = "INS"
                    gt = "1/1"
                elif pred == 3:  # Deletion (synthetic only)
                    ref, alt = self._get_deletion_alleles()
                    qual = int(confidence)
                    info_type = "DEL"
                    gt = "1/1"
                else:
                    continue

                depth = int(np.random.randint(20, 100))
                gq = int(confidence)

                line = (
                    f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\tPASS\t"
                    f"DP={depth};TYPE={info_type}\tGT:GQ\t{gt}:{gq}\n"
                )
                f.write(line)

        print(f"VCF file written to: {self.output_path}")

    def _get_snp_alleles(self):
        bases = ["A", "T", "G", "C"]
        ref = np.random.choice(bases)
        alt = np.random.choice([b for b in bases if b != ref])
        return ref, alt

    def _get_insertion_alleles(self):
        bases = ["A", "T", "G", "C"]
        ref = np.random.choice(bases)
        inserted = "".join(np.random.choice(bases, size=np.random.randint(1, 4)))
        alt = ref + inserted
        return ref, alt

    def _get_deletion_alleles(self):
        bases = ["A", "T", "G", "C"]
        deleted = "".join(np.random.choice(bases, size=np.random.randint(1, 4)))
        ref = np.random.choice(bases) + deleted
        alt = ref[0]
        return ref, alt


__all__ = ["VCFWriter"]