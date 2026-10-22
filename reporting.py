"""Human-readable reporting helpers for DeepCall predictions."""
from __future__ import annotations

DESCRIPTIONS = {
    0: "No mutation detected (reference sequence).",
    1: "Mutation detected: single DNA base changed (SNP).",
    2: "Mutation detected: extra bases inserted into the sequence (Insertion).",
    3: "Mutation detected: bases removed from the sequence (Deletion).",
}

REAL_DESCRIPTIONS = {
    0: "No variant (homozygous reference).",
    1: "Heterozygous variant (one chromosome carries the mutation).",
    2: "Homozygous alternate variant (both chromosomes carry the mutation).",
}


def describe_prediction(class_idx: int, confidence: float = None,
                       class_name: str = None, position=None,
                       ref: str = None, alt: str = None) -> str:
    """Return a plain-language description of a prediction."""
    if class_idx in DESCRIPTIONS:
        base = DESCRIPTIONS[class_idx]
    elif class_idx in REAL_DESCRIPTIONS:
        base = REAL_DESCRIPTIONS[class_idx]
    else:
        base = f"Predicted class {class_idx}."
    if class_name:
        base = f"{class_name}: {base}"
    parts = [base]
    if position is not None:
        parts.append(f"Position: {position}")
    if ref and alt and alt != ".":
        parts.append(f"Alleles: {ref} -> {alt}")
    if confidence is not None:
        parts.append(f"Confidence: {confidence:.2%}")
    return " | ".join(parts)


def summarize_variants(variants) -> str:
    """Summarize a list of variant dicts into a readable report."""
    if not variants:
        return "No variants detected in the scanned region."
    lines = [f"Found {len(variants)} variant(s):"]
    for v in variants:
        lines.append("  - " + describe_prediction(
            v.get("class", 0),
            confidence=v.get("confidence"),
            class_name=v.get("class_name"),
            position=v.get("pos"),
            ref=v.get("ref"),
            alt=v.get("alt"),
        ))
    return "\n".join(lines)