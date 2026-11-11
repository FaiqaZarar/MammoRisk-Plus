# MammoRisk+ — progress update for Dr. Mehwish Noureen (12 Nov 2026)

Team: Faiqa Zarar Noor (lead; Modules 1 and 3), Pukhraj Tahir (Module 2), Manaal Tufail (platform, evaluation).
Source for every line below: `docs/STATUS.md`, `docs/DECISIONS.md`, `CHANGELOG.md`, git history.

## 1. Where we are (honest summary)
All three modules, the shared evaluation library and the platform skeleton are **written and tested**
(79 automated tests pass, 34 commits). **No model results exist yet**, because the datasets are only
now being brought in. The next two weeks turn code into numbers.

| Module | Objective | State today | Next |
|---|---|---|---|
| Module 2 · PRS-313 | O1 (core) | Pipeline ready to run on Colab (GRCh38, gnomAD HGDP+1KG callset). Sample sheet built from real metadata: 394 people, **CEU 121, Pakistani pooled 273** (PJL 97 + 8 HGDP groups) | Pukhraj runs the Colab notebook; first coverage + % reclassified |
| Module 3 · classifier | O2 | Patient-level split made from the official CSVs: **3,032 images / 1,566 patients**, train/val/test 70/10/20, seed 42. Streaming cache builder tested; Kaggle training notebook written | Build cache on Kaggle, first EfficientNet-B3 vs ResNet-50 run |
| Module 3 · explanations | O3 | Grad-CAM/Score-CAM + Pointing Game, energy-in-box, IoU, sanity check coded and unit-tested. VinDr-Mammo access granted; real annotation files checked | 200-image VinDr sample downloading; run after first checkpoint |
| Module 1 · Mirai | O4 | Cohort builder, OncoServe client, C-index/td-AUC/calibration code written and tested | **Blocked on EMBED access** (section 2) |
| Platform | O5 | Flask app runs: hospital login, upload pages, PDF report. Every output must carry a validation label (enforced in code) | Wire in models as they appear; Docker still untested |
| Evaluation library | all | DeLong, bootstrap CIs, IPCW C-index, time-dependent AUC, calibration, XAI scores | — |

## 2. Blockers and what I need from you
1. **EMBED (Module 1, O4).** The open 20% release is a controlled-access bucket (an anonymous request returns
   403). My application of 29 July has had no reply. I am re-checking the HITI portal request and will send a
   short follow-up. **Request:** can you or Dr. Ilyas Ahmad add an institutional email to the follow-up, or
   do you know a contact at Emory? Fallback if it stays silent (needs your approval, would change O4):
   OPTIMAM (UK) or CSAW-CC (Sweden) applications, or Mirai on CMMD/DMID without follow-up (no C-index, only
   discrimination of biopsy-proven cancers).
2. **Objective numbering (D-11).** The revised review and the presented deck differ (O2–O4 order,
   ConvNeXt-Tiny yes/no, CMMD yes/no). Which version is final for the thesis?
3. **PLINK 2.0 only, or also pgsc_calc (D-10)?** And was Table 7 updated to the January release (D-06)?

## 3. Findings from real data this week (all counts, no images shown)
- **VinDr-Mammo** (explanation scoring, O3): 5,000 exams. Our rule (mass or suspicious calcification, BI-RADS ≥ 4)
  selects **788 images in 405 exams** (592 mass, 376 calcification). Scanners are Siemens, Planmed and
  IMS/Giotto, with **no Hologic**, so explanation scores will mix scanner and population effects.
- Download is slow (~72 KB/s from PhysioNet), so we take a **seeded, balanced sample of 200 images**
  (100 mass + 100 calcification, seed 42) and state it as a sample in the thesis (D-21).
- **Access issue solved:** PhysioNet refused `curl`/Python but accepted GNU `wget`; documented in D-21.
- Column names and category strings in the VinDr files match what our code expects.

## 4. Platform demo (runs on my laptop)
`python -m app.demo` → http://127.0.0.1:5055. The **About & validation** page shows what the platform
promises: each output (classifier, heatmap, Mirai risk, PRS under CEU and Pakistani references) lists the
datasets and populations it was validated on, and currently says **"Not yet validated"** until a real result
is logged. It also states "not validated on Pakistani women" explicitly. This is the reliability-labelling
idea of O5 working end to end, before any model is plugged in.

## 5. Plan to the app release (end of January 2027)
| When | What |
|---|---|
| This week | Module 2 Colab run; Kaggle cache + first training run; EMBED follow-up |
| Late Nov | First O1 numbers (% reclassified with bootstrap CIs, driver SNPs); classifier AUC on CBIS-DDSM test |
| Dec–Jan (exam buffer) | Mirai on EMBED if access arrives; explanations on CBIS-DDSM/DMID/VinDr; wire models into the platform; Docker |
| Feb–May | O1 paper first, then O3; thesis; final defence |

## 6. Risks
- Module 1 depends entirely on external data approval (fallbacks above).
- Small groups in Module 2 (8 HGDP groups of about 16–24): per-group results are descriptive only; the pooled
  Pakistani reference carries the headline.
- Dec–Jan includes exams, so slippage there has no buffer left.
