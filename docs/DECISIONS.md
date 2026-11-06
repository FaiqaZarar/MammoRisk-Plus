# Decision Log: MammoRisk+

Format: **D-xx · date · decision**, then alternatives, reason, decided by.
Never delete entries. Reversed decisions are marked `SUPERSEDED by D-xx`.
Source of truth for scope: the revised literature review (synopsis version) + the supervisor-revised defence deck.

---

## Scope and framing

### D-01 · Jul 2026 · Project retitled from PRS-MammoNet to MammoRisk+
- The old title was about fusing PRS with deep learning mammography. Fusion isn't feasible (D-02).
- Supervisor asked to drop "Validated" from the title. Final: *"MammoRisk+: Development and Validation of a
  Clinical Platform for Breast Cancer Risk Assessment and AI-Based Detection Using Deep Learning Mammography
  Analysis and Polygenic Risk Scoring"*. Title change form submitted.
- **Decided by:** Supervisor + team

### D-02 · Jul 2026 · Three independent modules, no fusion
- **Reason:** No public dataset pairs mammograms and genotypes from the same women. Azam 2026 and Arasu 2026
  had paired data, but it isn't public. Fusion is future work.
- Supervisor also asked to remove multimodal/fusion language and molecular subtypes (future work).
- **Decided by:** Supervisor + team

### D-03 · Sep–Oct 2026 · Research question reframed around Pakistani / South Asian women
- **Old framing (v1 review):** deployable tool with explainability that combines imaging and genomic risk.
- **New framing:** "European-derived and Western-trained risk tools are used without evidence that they
  classify Pakistani women correctly." The platform states its own reliability.
- **Reason:** The v1 gaps were weak. Clairity Breast was FDA-authorized in 2025, so "no tool exists" is false.
  The rubric's Innovation criterion rewards under-studied populations.
- Use **"South Asian or Pakistani"**, never just "Asian". Ho 2020/2022 covered Asian PRS, and Mirai was
  validated in Taiwan.
- **Decided by:** Team, for the synopsis defence

### D-04 · Oct 2026 · O1 (PRS reclassification across Pakistani groups) is the core contribution
- **Reason:** It needs no data approvals, it's novel (no study covers Pakistan's ethnic groups), and
  results come fastest.
- **Decided by:** Team

### D-05 · Jul 2026 · Public datasets only; Pakistani hospital data is future work
- Partner hospital datasets are kept **out of all official documents** (supervisor instruction).
- **Decided by:** Supervisor

### D-06 · Oct 2026 · MVP-first; app released end of January 2027
- The co-supervisor wants a working product. The platform is built Dec–Jan in parallel. Papers and thesis
  run Feb–May 2027.
- The defence deck showed "MVP 31 Oct, v1 7 Dec". The revised review Table 7 should say Platform Dec 2026–Jan
  2027. **Confirm Table 7 was updated.**
- **Decided by:** Team

## Module 1: Mirai

### D-07 · Oct 2026 · Pre-trained Mirai, external validation only, on EMBED (open subset)
- **Replaces v1:** "C-index on RSNA 2022 / CBIS-DDSM, 70/10/20 split".
- **Reason:** A C-index needs time-to-cancer follow-up. RSNA 2022 only has cancer status at screening, and
  CBIS-DDSM is film with no follow-up and no normals. We don't train Mirai, so there's no split.
- Report by density, race (Asian subgroup) and machine vendor. If EMBED has few events, results are exploratory.
- The v1 "target C-index 0.75–0.84" is dropped. The aim is to measure performance, not hit a target.
- **Decided by:** Team

### D-08 · Oct 2026 · Corrected the Mirai description
- The aggregator combines the **four views of one exam**. It is not a temporal aggregator across prior exams
  (v1 was wrong).
- Grad-CAM must load the Mirai repo + weights directly. OncoServe is for inference only.
- Dropped the Gail / Tyrer-Cuzick ablation (not feasible without questionnaire data). The 0.52/0.57 vs 0.76
  numbers are Park 2024's own model, not Mirai.

## Module 2: PRS

### D-09 · Oct 2026 · PRS populations and references
- **Replaces v1:** "validate on 1000G PJL, GIH, CEU".
- **Now:** 9 Pakistani groups = 1000G PJL (96) + 8 HGDP Pakistani groups (~20–25 each), ~280 pooled.
  Standardised against **European (CEU) vs Pakistani** references (own-group as a sensitivity analysis).
- Output: % reclassified, bootstrap CIs (stratified), driver SNPs. Top-1% only reported for the pooled sample.
- **Decided by:** Pukhraj / team

### D-10 · Oct 2026 · PLINK 2.0 (+ bcftools) instead of PRSice-2
- **Reason:** We apply published PGS000004 weights. PRSice-2 is for deriving or optimising PRS thresholds,
  which we don't need. PCA reference matching is used for ancestry.
- The defence deck mentioned pgsc_calc for allele matching. **Confirm: plink2 alone, or pgsc_calc?**

## Module 3: Classifier + explanations

### D-11 · Oct 2026 · Module 3 = benign vs malignant + explanation reliability, external datasets added
- **Replaces v1:** "suspicious vs normal on CBIS-DDSM, target AUC > 0.85".
- **Reason:** CBIS-DDSM has no normal images.
- **Now:** full mammograms (not crops), patient-level 70/10/20, ResNet-50 baseline vs EfficientNet-B3,
  target AUC ≥ 0.80. External transfer on DMID (+ CMMD). Explanations: Grad-CAM + Score-CAM scored with the
  Pointing Game, energy-in-box, IoU and a sanity check on CBIS-DDSM, VinDr-Mammo and DMID.
- ⚠️ **OPEN: review vs presented deck conflict.** The revised review had ResNet-50 + EfficientNet-B3, no CMMD,
  and objectives O2 = Mirai, O3 = explanations, O4 = platform. The supervisor-revised deck that was presented
  had ConvNeXt-Tiny, CMMD, and O2 = classifier, O3 = explanations, O4 = Mirai, O5 = platform.
  **Decide which is final and update the thesis to match.**
- **Decided by:** Team + supervisor (pending final confirmation)

## Platform

### D-12 · Oct 2026 · Flask + Docker, outputs carry validation labels
- **Alternatives:** FastAPI, Streamlit. Flask chosen for team familiarity, with Docker for reproducibility
  and to isolate Mirai's older dependency stack.
- Each output is stored with metadata: the datasets/populations it was tested on and its metric there.
  This metadata is printed in the report.

## Repository

### D-13 · 2026-11-06 · Repo restructured to the CLAUDE.md layout; fusion code removed; split environments
- **Layout:** `src/{imaging,genomics,preprocessing,evaluation,api}` + `frontend/` →
  `module1_mirai/`, `module2_prs/`, `module3_classifier/`, `evaluation/`, `app/` (templates + static), plus
  `splits/`. All moved packages were empty placeholders.
- **`src/fusion/` deleted.** Fusion is out of scope (D-02). Keeping a placeholder would suggest otherwise to
  the committee. Alternative: keep it empty. Rejected.
- **Legacy PRS pilot:** `genomics/scripts/` + `genomics/docs/PRS_Pipeline.md` moved **unchanged** to
  `module2_prs/legacy_prsice/`. It is Pukhraj's chr1 PRSice-2 pilot, frozen for his review and not to be
  rewritten by others. The new pipeline uses plink2 (D-10).
- **Environments:** `requirements.txt` → `environment-imaging.yml` (Windows + CUDA: torch, timm,
  pytorch-grad-cam, pydicom, scikit-survival, lifelines, Flask, wandb; shap dropped as unused) and
  `environment-genomics.yml` (macOS: plink2 + bcftools from bioconda). Reason: the imaging and genomics
  work run on different machines (CLAUDE.md compute constraint), and bioconda has no Windows builds. On
  Windows, plink2 is used as a standalone binary.
- **PDFs not committed:** `docs/*.pdf` (literature reviews, defence deck) are gitignored and kept locally.
- **PRS weights:** `data/prs/PGS000004.txt.gz` stays committed as a small public file. The unpacked
  duplicate was removed. ⚠️ Its header says `#genome_build=GRCh37` (author-reported, **not** the GRCh38
  harmonised file CLAUDE.md specifies). It has not been replaced yet. See open items.
- **Decided by:** Faiqa

---

## Deviations from the FIRST (July) literature review
| # | July review said | Now | Decision |
|---|---|---|---|
| 1 | Gaps: no deployable tool / no integration / no XAI | Gap: no evidence for Pakistani women | D-03 |
| 2 | Mirai "temporal aggregator" | Four-view aggregator for a single exam | D-08 |
| 3 | Mirai C-index on RSNA 2022 + CBIS-DDSM, 70/10/20 split | EMBED open subset, no split, external validation | D-07 |
| 4 | Target C-index 0.75–0.84 | No target; report by subgroup | D-07 |
| 5 | Gail/Tyrer-Cuzick ablation | Dropped | D-08 |
| 6 | PRS validated on PJL, GIH, CEU | 9 Pakistani groups (PJL + 8 HGDP) vs CEU | D-09 |
| 7 | PRSice-2 + PLINK 2.0 | PLINK 2.0 + bcftools (± pgsc_calc) | D-10 |
| 8 | M3 "suspicious vs normal", AUC > 0.85 | Benign vs malignant, AUC ≥ 0.80 | D-11 |
| 9 | Grad-CAM only, visual inspection | Grad-CAM + Score-CAM, quantitative scores | D-11 |
| 10 | Datasets: CBIS-DDSM, RSNA, EMBED, 1000G | + VinDr-Mammo, DMID, HGDP, (CMMD); RSNA dropped | D-07, D-11 |
| 11 | Platform/paper Dec 2026–Jan 2027 | App Jan 2027; papers + thesis Feb–May 2027 | D-06 |

## Open items to confirm
- [ ] D-11: final objective numbering, ConvNeXt yes/no, CMMD yes/no
- [ ] D-10: pgsc_calc or plink2 only
- [ ] D-06: Table 7 in the review updated to the January release?
- [ ] D-13: committed PGS000004 is GRCh37 (author-reported). Use the GRCh38 harmonised file (CLAUDE.md) or
  keep GRCh37? It must match the genotype build (1000G phase 3 = GRCh37; HGDP/1kG high-coverage = GRCh38).
- [ ] D-13: Pukhraj to review `module2_prs/legacy_prsice/`
- [ ] Any feedback from the synopsis defence panel, which should be recorded here as D-14 onwards
