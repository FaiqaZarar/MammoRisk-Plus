import pandas as pd

# ----------------------------------------
# Read the matched variants
# ----------------------------------------
input_file = r"C:\MammoRiskPlus\results\matched_variants.csv"

df = pd.read_csv(input_file)

# ----------------------------------------
# Select and rename the columns for PRSice
# ----------------------------------------
prsice = df[[
    "rsid",
    "chr",
    "position",
    "effect_allele",
    "other_allele",
    "effect_weight"
]].copy()

prsice.columns = [
    "SNP",
    "CHR",
    "BP",
    "A1",
    "A2",
    "BETA"
]

# ----------------------------------------
# Add a dummy P-value column
# ----------------------------------------
prsice["P"] = 1

# ----------------------------------------
# Save the new file
# ----------------------------------------
output_file = r"C:\MammoRiskPlus\results\prsice_base.txt"

prsice.to_csv(
    output_file,
    sep="\t",
    index=False
)

# ----------------------------------------
# Print summary
# ----------------------------------------
print("=" * 50)
print("PRSice BASE FILE CREATED")
print("=" * 50)

print(f"Variants : {len(prsice)}")
print(f"Saved to : {output_file}")