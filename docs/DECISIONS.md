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

## Code build-out (2026-11-06). Proposed while Pukhraj and Manaal were unavailable

Entries D-14 to D-17 were made by Faiqa (with Claude Code) so the work wasn't blocked. Each is marked
**PROPOSED** where the module owner still has to confirm it.

### D-14 · 2026-11-06 · Module 2 methods details (PROPOSED, needs Pukhraj's review)
- **Risk categories** on the reference percentile scale (fitted normal): low < 20th, average 20th–80th,
  elevated 80th–99th, high ≥ 99th (top 1%). These follow the percentile bands Mavaddat 2019 reports.
  Config-driven (`configs/m2_prs.yaml`). Alternatives: deciles only, or Mavaddat's 11 bands (too many
  for groups of ~22 people).
- **Palindromic A/T, C/G SNPs (31 of 313):** `keep`, assuming the forward strand, because the PGS
  Catalog, 1000G and HGDP all report forward-strand alleles. `drop` is a sensitivity analysis.
  Resolving them by allele frequency was **rejected**: PGS000004 frequencies are European, and
  frequency differences between populations are exactly what O1 measures.
- **Indels (48 of 313):** exact allele match after `bcftools norm`, never complemented.
- **Bootstrap:** each resample re-draws CEU and all 9 groups (stratified by group) and **re-fits
  both references**, so the CI includes uncertainty in the reference mean/SD, not only in the targets.
- **Raw score = plink2 SUM** (`cols=+scoresums`), not plink's default per-allele average.
- **Data:** extract only the 313 positions from the remote tabix-indexed VCFs (`bcftools view -R`),
  which is megabytes, not whole-genome downloads. PCA still needs genome-wide common SNPs.
- **Driver SNPs:** exact decomposition Δmean = Σ 2β_j(p_PAK − p_CEU), in CEU-SD units.
- **Decided by:** Faiqa (proposed). **Pukhraj to confirm.**

### D-15 · 2026-11-06 · Module 1 cohort and metric rules (PROPOSED)
- C-index = Uno's IPCW version, τ = 5 y, ranked on Mirai's 5-year risk (as in Yala 2021/2022). Harrell's
  C is also reported. td-AUC at year t uses Mirai's t-year risk.
- All of a woman's screening exams are kept, and CIs are **clustered by patient** (1,000
  resamples). Alternative: one random exam per woman (loses events).
- Exams after a woman's first cancer are excluded. Same-day cancers are events at t = 0.5 days.
  `min_days_to_event` (e.g. 180) is a sensitivity analysis that tests prediction rather than detection.
- Censoring is at the woman's last exam of any type. Duplicate view images: keep the first by
  sorted path.
- Subgroups with fewer than 5 cancers: counts only, flagged exploratory, no CI.
- Vendor: Hologic / Other / Mixed. **EMBED "Asian" is not South Asian.** Say so wherever it's reported.
- EMBED column names and the OncoServe API are **TO VERIFY** on the real files (`COLUMNS` dict, config).
- **Decided by:** Faiqa (proposed)

### D-16 · 2026-11-06 · Platform design (PROPOSED, needs Manaal's review)
- Validation labels are enforced in code: no result can be shown or printed without a label, and a
  label can only say "validated" with a value + source (which must trace to STATUS.md).
- Adapters say "model not available" instead of showing placeholder numbers.
- A minimal session login (werkzeug hashes from env) and custom CSRF, with no Flask-Login/WTF
  dependency. Results are kept in the session cookie for the PDF (to move server-side once heatmaps
  are shown). The PDF uses fpdf2. Uploads go to tmp/tmpfs and are deleted after each request.
  Weights are mounted read-only, never baked into the image.
- **Decided by:** Faiqa (proposed). **Manaal to confirm** (Flask-Login/DB vs current).

### D-18 · 2026-11-06 · Module 3 implementation details (PROPOSED)
- CBIS-DDSM's official train/test split is **not used**. Mass + calc are merged by patient, then split
  70/10/20 at patient level, stratified by patient label (CLAUDE.md).
- Preprocessing: all breasts are flipped to face left, padded to keep the aspect ratio at ~1024×512,
  and saved as 16-bit PNGs. Masks get the same transforms. Horizontal-flip augmentation is kept (as
  CLAUDE.md specifies), even though it undoes the canonical orientation for that sample.
- Training: balanced class weights, AdamW, cosine schedule with warm-up, gradient accumulation
  (effective batch 32), AMP.
- Test AUC is reported with **both** a DeLong CI and a patient-clustered bootstrap CI.
- CAM target layer = last feature map before pooling (`bn2` EfficientNet-B3, `layer4[-1]` ResNet-50).
  The sanity check uses Grad-CAM only (Score-CAM is too slow for layer-by-layer randomisation).
- VinDr-Mammo has no biopsy labels, so for explanation scoring "malignant" = a Mass or Suspicious
  Calcification finding with BI-RADS ≥ 4. VinDr images/overlays are never exported. DMID normals are
  excluded from the AUC.
- Dataset column names (CBIS, CMMD, DMID, VinDr) are **TO VERIFY** on the real files.
- **Decided by:** Faiqa (proposed). Manaal supports Module 3.

### D-17 · 2026-11-06 · Data storage and local test environment
- Raw datasets live on an external **USB drive (NTFS/exFAT), not OneDrive**. OneDrive on-demand files
  get downloaded back onto the laptop when read, and it is slow with many small DICOMs. Every config has
  a data path or `data_root`, so changing the drive means editing the config only.
- CBIS-DDSM training on Kaggle/Colab. Only the downsampled PNG cache is kept locally.
- Local tests run in a project `.venv` (Python 3.13, gitignored) because conda/torch aren't on this
  laptop. scikit-survival is installed with `--no-deps`, since its `ecos` solver doesn't build on
  Windows/3.13 and only the metrics are used. The conda env files remain the reference environments.
- **Decided by:** Faiqa

### D-19 · 2026-11-07 · Module 2 genotypes = gnomAD v3.1.2 HGDP+1KG callset on GRCh38
- **Decision:** take CEU, PJL and the 8 HGDP Pakistani groups from the **jointly called gnomAD v3.1.2
  HGDP+1KG subset (GRCh38)**, and score with the **PGS000004 GRCh38 harmonised** file. This resolves the
  D-13 open item (GRCh37 vs GRCh38) in favour of CLAUDE.md's GRCh38.
- **Alternatives:** (a) 1000G phase 3 (GRCh37) + Bergström 2020 HGDP (GRCh38), which would mean two
  builds and two calling pipelines; (b) 1000G 30x NYGC + Bergström HGDP, the same build but still two
  pipelines.
- **Reason:** with separately called datasets, part of the CEU vs Pakistani PRS difference could be a
  **batch effect** (different sequencing and variant calling), which would confound O1. One joint
  callset removes that. It is public, and tabix-indexed per chromosome, so `bcftools view -R` fetches
  only the 313 positions out of 3.63 TB.
- **Samples:** gnomAD `high_quality` and `related` flags applied. Result (2026-11-07): CEU 121, PJL 97,
  Balochi 23, Brahui 23, Burusho 24, Hazara 16, Kalash 21, Makrani 22, Pathan 24, Sindhi 23. That gives
  **Pakistani pooled 273**, 394 in total. Note: CEU is 121 rather than 1000G phase 3's 99, because the 30x
  release adds samples. Counts differ slightly from CLAUDE.md (PJL 96 → 97).
- **Both sexes are used.** PRS-313 is entirely autosomal, and the HGDP groups are mostly men (women
  only would leave ~4 per HGDP group). This is a deviation worth stating in the thesis.
- **Ancestry check:** gnomAD's genome-wide PCA scores + its subcontinental outlier flag, plus our >4 SD
  within-group flag on the first 2 PCs, so no genome-wide download is needed. Flagged people are kept in the
  main analysis and excluded in a sensitivity analysis (0 flagged on 2026-11-07).
- **Variant QC:** only gnomAD PASS sites. A failing PRS SNP counts as missing (2·EAF) and lowers coverage.
- **Where it runs:** Colab (`notebooks/m2_prs_colab.ipynb`): Ubuntu bcftools + plink2 Beta 1.3
  (2026-10-31, generic Linux build).
- **Decided by:** Faiqa (with Claude Code), 2026-11-07. Pukhraj to review.

### D-20 · 2026-11-09 · CBIS-DDSM streamed from the TCIA API into the PNG cache (no raw DICOM storage)
- **Decision:** `module3_classifier/fetch_cbis.py` downloads one series at a time from the public TCIA
  NBIA API, writes the 16-bit PNG + mask, and deletes the DICOM at once. It runs on Kaggle
  (`notebooks/m3_cbis_cache_kaggle.ipynb`), and the cache becomes a **private** Kaggle dataset for training.
- **Alternatives:** (a) NBIA Data Retriever → 163.6 GB on a USB drive → `preprocess.py` (D-17 plan),
  which needs the full disk space and a separate deletion step; (b) the Kaggle JPEG mirror of CBIS-DDSM,
  which is 8-bit lossy and a third-party re-encoding with unclear provenance.
- **Reason:** peak disk is a few hundred MB instead of 164 GB. Data come straight from the source, at
  full bit depth, with the same preprocessing code. The run is resumable and logs every image.
- **Masks:** all ROI series of a lesion are fetched (TCIA PatientID = lesion folder), and the mask is the
  file with the full-image shape and two values. This handles the ~107 series where TCIA labels the mask
  series "cropped images".
- **Split created 2026-11-09** from the official TCIA description CSVs (before any image was seen):
  `splits/m3_cbis_split.csv`, 3,032 full mammograms / 1,566 patients, 44% malignant images. Train 1,095 /
  val 157 / test 314 patients (2,138 / 291 / 603 images), seed 42.
- **Checked:** a trial on 3 images gave 3/3 ok with masks, and a visual check showed correct orientation
  and mask alignment. Home bandwidth gives ~3 img/min (~17 h for all), hence Kaggle.
- **Supersedes** the CBIS part of D-17 (USB for raw CBIS). USB is still used for VinDr/CMMD.
- **Decided by:** Faiqa (with Claude Code)

### D-21 · 2026-11-11 · VinDr-Mammo: download via wget, and only a seeded sample (not all 788 images) at first
- **Decision:** (1) VinDr files are downloaded with the real GNU wget (inside WSL) from a URL list written by
  `module3_classifier/fetch_vindr.py`, which also selects the images and verifies SHA256. (2) The first
  download is a **seeded sample of 200 images (100 mass + 100 calcification; 170 exams; ~3.4 GB)** drawn from
  the 788 D-18 images, `--sample 200 --seed 42`, balanced and ordered so every prefix is balanced. The rest
  can be fetched later.
- **Why wget:** with a valid login and signed DUA, PhysioNet's file server returned **403 to curl.exe and to
  Python urllib**, but accepted GNU wget (401 challenge, then 200). Cause not confirmed by PhysioNet
  documentation (a third-party source says the server only serves clients identifying as Wget). We use the
  documented client and do not spoof anything. winget's wget package failed its hash check, so wget from
  WSL Ubuntu is used.
- **Why a sample:** measured speed ~140 KB/s, so 788 images (~13 GB) would take about a day. Explanation
  scores (Pointing Game etc.) need a sample, not the whole set. Alternative: full subset overnight.
- **Effect on results:** VinDr explanation scores come from 200 of 788 eligible images. Report N, the seed,
  the mass/calcification split, and that it is a sample. Vendors in VinDr are Siemens/Planmed/IMS (no Hologic).
- **Licence:** files stay on `D:\mammorisk_data\vindr_mammo\` only. The sample list lives next to the data
  (`vindr_sample_n200_seed42.csv`), reproducible from the CSVs + seed. Nothing is uploaded or committed.
- **Decided by:** Faiqa (with Claude Code)

### D-22 · 2026-11-12 · Platform frontend redesign, results pages, evidence page, stamped demo mode (PROPOSED, needs Manaal's review)
- **Decision:** redesign the Flask frontend (D-16 skeleton) as a proper application: design system with
  light/dark themes, app shell with sidebar after login, landing page, a **reliability matrix** (outputs x
  regions: USA, China, India, Vietnam, Europe, Pakistan), result pages with figures (probability bar, Mirai
  1-5 year bars, PRS reference curves with the D-14 risk bands), an Evidence page (matrix, labels, datasets
  and limits, methods), a dashboard with model status, this-session results and the project schedule, and
  styled error pages.
- **Constraints kept:** no new Python dependencies; no JavaScript libraries, CDN or web fonts (inline SVG,
  system fonts, ~60 lines of optional vanilla JS), so it runs offline in a hospital. Every result still needs
  a validation label (D-16); the matrix cannot overstate: a region shows "validated" only when the label has a
  value and source. Label schema gained optional `short` and `regions` fields (`app/validation_labels.yaml`).
- **Demo mode (stamped illustrative output):** off by default; enabled only by `MAMMORISK_DEMO=1` or
  `python -m app.demo`, never in Docker. When a model is missing it returns fixed sample numbers, each flagged
  `illustrative`, stamped "Illustrative, not real data" on the page and in the PDF, with a banner; the label
  still says "Not yet validated". The numbers are NOT computed from uploaded files. Alternative rejected:
  empty states only (safer but nothing to show at the supervisor meeting). **Decided by:** Faiqa (asked).
- **Project schedule/status on the dashboard** reads the module table from `docs/STATUS.md` at run time
  (hidden if the file is missing, as in the Docker image, which ships only `app/`) and the revised timeline
  from `app/project_status.py`.
- **Open for Manaal:** wording of the pages, the login approach (D-16), whether `docs/STATUS.md` should be
  copied into the image, and wiring real model outputs into the figures (adapters must return the keys the
  figures read: `probability`/`threshold`, `1-year`..`5-year`, `z`/`percentile`).
- **Decided by:** Faiqa (with Claude Code). **Manaal to confirm.**

### D-23 · 2026-11-12 · VinDr cache: same preprocessing as CBIS-DDSM, local only, boxes checked numerically
- **Decision:** `module3_classifier/cache_vindr.py` turns the downloaded VinDr DICOMs into the 1024x512
  16-bit PNG cache plus an index with lesion boxes in cached coordinates, using exactly the CBIS-DDSM
  preprocessing (percentile normalisation, canonical left orientation, aspect-preserving resize and pad).
  Reason: a score difference between datasets must not come from preprocessing.
- **Safeguards:** an image whose DICOM size differs from the annotation's height/width is refused (the boxes
  would be misplaced); failed or not-yet-downloaded images are recorded in the index and skipped by the
  explanation code (`status == ok` only); the output folder must be outside the repo or under the gitignored
  `data/` (licence: VinDr images are never committed, uploaded or shown).
- **Check done (2026-11-12, numbers only, no images displayed).** First on 15 real images, then on the complete
  sample (`vindr_sample_n200_seed42`, all 200 images downloaded, SHA256-verified and cached). Full-sample
  results: every one of the 200 breasts ends up on the left; 91 images were flipped (90 are right-breast
  labels, plus one left-breast image whose raw file has the breast on the right, which the flip corrected);
  the real boxes have higher mean intensity than the rest of the tissue in 191 of 200 images (mean 0.744 vs
  0.531) and exceed the 75th percentile of random same-size boxes in 199 of 200. Three of 247 boxes overshot the
  image edge in the source annotations (by 0.2 to 27 original pixels); they are now clipped to the image, so
  all 247 are valid. This supports correct box placement after flip and scale. It is not proof for every
  image, and the 9 images where the box is not brighter are small or low-contrast lesions, to be remembered
  when reading the explanation scores.
- **Alternative rejected:** checking by looking at overlays (VinDr licence forbids showing images).
- **Decided by:** Faiqa (with Claude Code)

### D-24 · 2026-11-12 · CPU-only torch in the laptop .venv, for smoke tests only (amends D-17)
- **Decision:** installed CPU torch 2.15, torchvision, timm and grad-cam (without its own OpenCV; the project
  already has `opencv-python-headless`) into the project `.venv`. Purpose: run the real train, evaluate and
  explain code end to end before spending Kaggle GPU hours. No training of real models happens on the laptop.
- **Smoke test (synthetic data, 96 tiny images, random weights; the numbers mean nothing):** EfficientNet-B3
  and ResNet-50 trained 6 epochs each (about 100 s per run on CPU); `evaluate run` gave DeLong and bootstrap
  CIs, the val-chosen threshold and predictions; `evaluate compare` gave the paired DeLong test; `explain`
  ran Grad-CAM, Score-CAM and the weight-randomisation check (Score-CAM about 10 min on CPU for 8 images).
  The real VinDr cache (6 images, Grad-CAM) also ran, and with `save_overlays` requested no overlay, `.npy` or
  image was written (licence guard works). ImageNet weights load for both backbones.
- **Not verified:** GPU and AMP behaviour, speed on 1024x512 inputs, and any real model quality. The first
  Kaggle run is still the real test.
- **Decided by:** Faiqa (approved the install), with Claude Code.

### D-25 · 2026-11-12 · IoU on the top 50% of pixels: ties at the threshold are excluded (fixes a bug)
- **Problem found by the smoke test:** `IoU` used `heatmap >= median`. A Grad-CAM is ReLU'd, so after
  normalisation a large block of pixels is exactly 0. When more than 50% are zero the median is 0 and every
  pixel was selected, so IoU equalled the lesion's area fraction (the "area baseline") whatever the heatmap
  showed. This would have silently corrupted the IoU results of O3.
- **Decision:** when ties at the threshold push the selection beyond the requested fraction (more than 1%
  slack), keep only pixels strictly above the threshold. So "top 50%" means the pixels the method actually
  highlights, up to 50%. An all-zero heatmap scores IoU 0. The fraction really selected is now reported
  (`iou_selected_fraction`) next to the threshold, per method, in the results.
- **Alternatives rejected:** ranking pixels and picking exactly 50% (would add arbitrary zero-valued pixels,
  so random placement of those would add noise); changing the top fraction (the 50% is CLAUDE.md's spec).
- **Effect on the smoke test:** Grad-CAM IoU went from 0.0188 (exactly the baseline) to 0.0445, with 25% of
  pixels selected. Three regression tests added (`tests/test_evaluation.py`).
- **Thesis note:** report the selected fraction with the IoU, since it varies by method (a sparse heatmap
  selects fewer than 50% of pixels).
- **Decided by:** Faiqa (with Claude Code)

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
| 12 | Genotypes: 1000G PJL (96) + CEU, HGDP separately | One jointly called gnomAD HGDP+1KG GRCh38 callset; CEU 121, PJL 97, HGDP 176; both sexes | D-19 |

## Open items to confirm
- [ ] D-11: final objective numbering, ConvNeXt yes/no, CMMD yes/no
- [ ] D-10: pgsc_calc or plink2 only
- [ ] D-06: Table 7 in the review updated to the January release?
- [x] D-13: GRCh37 vs GRCh38. Resolved by D-19: GRCh38 harmonised weights + gnomAD GRCh38 callset.
- [ ] D-13: Pukhraj to review `module2_prs/legacy_prsice/`
- [ ] D-14: Pukhraj to confirm risk categories, palindromic policy, bootstrap design
- [ ] D-15: verify EMBED column names + OncoServe API on real files
- [ ] D-16: Manaal to confirm login/CSRF approach and UI wording
- [ ] D-18: verify CBIS-DDSM / CMMD / DMID / VinDr column names; first torch run on Kaggle/Colab.
  2026-11-11 (VinDr, from the PhysioNet data description, not the files): `finding_annotations.csv` columns
  match `load_vindr_boxes` (`study_id, image_id, laterality, view_position, height, width, finding_categories,
  finding_birads, xmin..ymax`). VinDr access needs only the signed DUA (Restricted licence); no
  credentialing/CITI. Only BI-RADS 3-5 findings are boxed.
  **Checked on the real CSVs 2026-11-11** (`D:\mammorisk_data\vindr_mammo\`): columns match; the
  calcification string is exactly "Suspicious Calcification"; `finding_annotations.csv` also has "No Finding"
  rows (NaN boxes, NaN BI-RADS; dropped by `dropna`). D-18 rule (Mass or Suspicious Calcification, BI-RADS >= 4)
  selects 1,039 boxes on **788 images in 405 exams** (592 mass, 376 calc images), about 13 GB of the 337.8 GB
  (27 GB if all 4 views of those exams are kept). Only the subset is downloaded (D: has 212 GB free).
  Vendors: Siemens 15,244 images, Planmed 3,796, IMS/Giotto 960 (no Hologic), so VinDr is a different vendor
  mix from CBIS-DDSM; state this when interpreting explanation scores.
- [ ] Any feedback from the synopsis defence panel, which should be recorded here as the next free D-number
