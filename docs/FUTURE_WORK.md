# Future work and extension ideas: MammoRisk+

Nothing here is decided or in scope. The revised literature review stays the source of truth for the project.
An item moves into the project only after the supervisor agrees and a deviation is logged in `docs/DECISIONS.md`.

## A. Already agreed as future work
- **Fusion of imaging and genetic risk.** Needs a dataset with mammograms and genotypes from the same women; none is
  public (D-02).
- **Validation on Pakistani hospital mammograms.** Not a dependency; public data only for now (D-05).
- **Public Urdu awareness app.** The current platform is the hospital-facing one (CLAUDE.md, O5).
- **Gail and Tyrer-Cuzick comparison.** Needs questionnaire inputs we do not have (D-08).
- **Molecular subtypes.** Removed from scope at the supervisor's request (D-02).

## B. Extension ideas from the literature check (2026-11-12)
Source: a limited web search, read at abstract or snippet level. "No study found" means none turned up in that
search, not that none exists. Before any novelty claim goes into the thesis, read the papers in section C.

### B1. Reclassification versus genetic distance (Module 2, O1)
- **Idea:** compare three ways of standardising PRS-313: against a European (CEU) reference, against the pooled
  Pakistani reference, and with a continuous ancestry adjustment (regressing the score on principal components).
  Plot reclassification and the z-score shift against each person's genetic distance from the CEU reference, and
  measure agreement between the three methods.
- **Why it adds depth:** turns "X% of people change category" into a rule for how far from the reference the
  category flips. Earlier work standardised against European controls and reported no difference in discrimination
  across Asian groups (Ho et al. 2020); we found no study of category changes or of Pakistani groups.
- **Cost and fit:** low. Uses the gnomAD principal-component scores already in D-19, so no new data. It fits inside O1.
- **Cautions:** the 8 HGDP groups have about 16 to 24 people each, so group-level results stay descriptive.

### B2. Case-level reliability flag (platform, Module 3)
- **Idea:** the platform shows how similar an uploaded image is to the data the model was validated on (distance in
  the classifier's feature space). Test on CMMD, DMID and VinDr whether that distance predicts the drop in
  accuracy.
- **Why it adds depth:** the project's claim is that the platform states its own reliability. Model cards are mostly
  static documents; this makes the statement per case. It is the AI-focused extension for the co-supervisor.
- **Cost and fit:** medium. Needs a trained model and the external caches. Extends O3 and O5.

### B3. Do explanations fail before accuracy does? (Module 3, O3)
- **Idea:** relate localisation scores (Pointing Game, energy-in-box, IoU) to accuracy and to the distance from B2
  across datasets. Add a quick test of how well the dataset of origin can be predicted from the images or the
  model's features, to measure the scanner-versus-population confound instead of only stating it.
- **Why it adds depth:** the known limitation "machine type and population are confounded" becomes a measurement.
- **Cost and fit:** medium. Needs trained model and cached external sets. Extends O3.

### B4. Calibration and threshold transfer (Module 3, O2)
- **Idea:** report calibration (reliability curve, expected calibration error) on each external set. Show how
  sensitivity and specificity move when the CBIS-DDSM threshold is applied to CMMD and DMID. Optionally test
  recalibration on a small target subset.
- **Why it adds depth:** separates a ranking problem (AUC) from a threshold problem, which is what a hospital feels.
- **Cost and fit:** low. Uses the existing evaluation library.

### What not to add
Fusion models, new architectures, or chasing a higher AUC. The contribution is measuring how far tools can be
trusted and saying so on every output.

### How to keep the work credible
- Write the analysis plan before seeing PRS results (what is compared, what counts as reclassified, which
  sensitivity analyses). Report null results.
- State the sample sizes of every subgroup, and say "we found no study" rather than "first".
- The December and January exams leave little time: finish the agreed scope first, then take B1, then B4.

## C. Read before claiming novelty
- Ho et al., PRS-313 in Asian women, *Nature Communications* 2020, and Ho et al. 2022 (PRS plus non-genetic factors in Asian women).
- Mavaddat et al., PRS-313, *AJHG* 2019.
- Yala et al., Mirai, *Science Translational Medicine* 2021, and the multi-institutional validation (*JCO* 2022).
- "Dataset-Origin Signatures and Shortcut Learning in Screening Mammography AI" (arXiv 2607.15416, 2026): directly relevant to B3.
- Cerekci et al. 2024 (Pointing Game for saliency in mammography) and Mellado et al. 2025 (CorRELAX on VinDr-Mammo, *Frontiers in Oncology*).
- PGS Catalog guidance on reporting scores across genetic ancestry (pgsc_calc documentation).

## Sources found in the 2026-11-12 search
- [Polygenic risk scores for breast cancer in Asian populations](https://pubmed.ncbi.nlm.nih.gov/34906514/)
- [PGS Catalog: reporting and adjusting PGS across genetic ancestry](https://pgsc-calc.readthedocs.io/en/dev/explanation/geneticancestry.html)
- [Mirai multi-institutional validation](https://pubmed.ncbi.nlm.nih.gov/34767469/)
- [Pointing Game for saliency methods in mammography](https://bohrium.dp.tech/paper/arxiv/38786433)
- [CorRELAX on VinDr-Mammo](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1601929/epub)
- [Dataset-origin shortcut learning in mammography](https://arxiv.org/pdf/2607.15416)
- [Domain generalization in mammography mass detection](https://arxiv.org/pdf/2201.11620)
- [Model cards for healthcare](https://nabla.com/blog/what-is-a-model-card-a-primer-for-healthcare-leaders)
