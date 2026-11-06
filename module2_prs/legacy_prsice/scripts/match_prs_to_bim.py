import pandas as pd

# -----------------------------
# Load the PRS chromosome 1 file
# -----------------------------
prs = pd.read_csv(
    r"C:\MammoRiskPlus\results\chr1_prs_variants.csv"
)

# -----------------------------
# Load the PLINK BIM file
# -----------------------------
bim = pd.read_csv(
    r"C:\MammoRiskPlus\data\genotypes\chr1_phase3.bim",
    sep=r"\s+",
    header=None,
    names=[
        "chr",
        "rsid",
        "genetic_distance",
        "position",
        "allele1",
        "allele2"
    ]
)

# -----------------------------
# Match by chromosome + position
# -----------------------------
matched = prs.merge(
    bim,
    left_on=["chr_name", "chr_position"],
    right_on=["chr", "position"],
    how="inner"
)

# -----------------------------
# Find missing variants
# -----------------------------
missing = prs.merge(
    bim,
    left_on=["chr_name", "chr_position"],
    right_on=["chr", "position"],
    how="left",
    indicator=True
)

missing = missing[missing["_merge"] == "left_only"]

# -----------------------------
# Save results
# -----------------------------
matched.to_csv(
    r"C:\MammoRiskPlus\results\matched_variants.csv",
    index=False
)

missing.to_csv(
    r"C:\MammoRiskPlus\results\missing_variants.csv",
    index=False
)

# -----------------------------
# Print summary
# -----------------------------
print("=" * 50)
print("MATCH SUMMARY")
print("=" * 50)

print(f"PRS chromosome 1 variants : {len(prs)}")
print(f"Matched variants          : {len(matched)}")
print(f"Missing variants          : {len(missing)}")

print("\nFiles created:")
print("matched_variants.csv")
print("missing_variants.csv")