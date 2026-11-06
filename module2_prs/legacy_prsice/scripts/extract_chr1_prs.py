import pandas as pd

# Input PRS file
input_file = r"C:\MammoRiskPlus\data\prs\PGS000004.txt\PGS000004.txt"

# Read the PRS scoring file
prs = pd.read_csv(
    input_file,
    sep="\t",
    comment="#"
)

# Keep only chromosome 1 variants
chr1 = prs[prs["chr_name"] == 1]

# Save the chromosome 1 variants
output_file = r"C:\MammoRiskPlus\results\chr1_prs_variants.csv"
chr1.to_csv(output_file, index=False)

# Print summary
print("=" * 40)
print("PRS SUMMARY")
print("=" * 40)
print(f"Total variants in PRS : {len(prs)}")
print(f"Chromosome 1 variants : {len(chr1)}")
print(f"Output saved to       : {output_file}")