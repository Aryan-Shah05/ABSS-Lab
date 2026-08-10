 # Biometric Authentication Evaluation — Assignment 1

This repository contains a complete implementation for the assignment "User Authentication using Biometric Features".
It loads the provided `biomet_data.csv` (feature vectors for 100 users, 10 samples per user, 144 features per sample),
creates enrollment templates, compares test samples against templates using two matching criteria (Euclidean distance and cosine similarity),
and computes the standard biometric performance measures required by the assignment:

- Genuine and imposter score distributions
- FAR and FRR vs threshold
- ROC curve (TPR vs FPR)
- Equal Error Rate (EER)
- Decidability index (d')

This README documents assumptions, implementation details, how to reproduce the results, and guidance for interpretation.

## 1) Data assumptions

- `biomet_data.csv` should contain 1000 samples × 144 feature values per sample (total 144000 numbers). The ordering expected by the code is:
	- Users in sequence 0..99
	- For each user: 10 samples in order (samples 0..4 are enrollment, 5..9 are test samples taken 6 months later)
	- Row-major layout where each row corresponds to one sample and contains 144 whitespace-separated feature values. The loader accepts standard whitespace-delimited numeric text.

If your file is laid out differently (for example, wrapped lines, different row/column orientation), the loader will raise an informative error — contact me and I will adapt the loader.

## 2) How the system is built (implementation details)

- Template creation: For each user, the enrollment template is the element-wise average of the first five feature vectors for that user.
- Comparisons: For each test sample (5 tests × 100 users = 500 test samples), the code compares the test vector with all 100 templates, producing 100 matching scores per test. That yields 50,000 comparisons.

Matching metrics implemented:
- Euclidean distance: computed as the L2 norm between template and test. To treat higher=better (so distributions align with cosine similarity which is higher for good matches), the code negates distances and uses negative distance as the score.
- Cosine similarity: normalized dot product between template and test (range roughly -1..1). Higher values indicate closer matches.

## 3) Genuine vs Imposter labeling

- Genuine scores: for each test sample of user u, the score with template u is collected (500 genuine scores total).
- Imposter scores: for the same test sample, the scores with all other users' templates (99 per test) are collected (49500 imposter scores total).

## 4) Threshold-based metrics

- For a score threshold τ, the system accepts if score >= τ.
- FAR (False Acceptance Rate) at τ: fraction of imposter scores >= τ (imposters accepted).
- FRR (False Rejection Rate) at τ: fraction of genuine scores < τ (genuine rejected).

## 5) EER and Decidability

- EER: the threshold where FAR and FRR are closest; reported as the average of FAR and FRR at that operating point.
- Decidability index (d') is computed as

$$
d' = \frac{|\mu_g - \mu_i|}{\sqrt{\tfrac{1}{2}(\sigma_g^2 + \sigma_i^2)}}
$$

where μ_g, μ_i are the genuine and imposter means and σ_g, σ_i are their standard deviations. Higher d' indicates better separability.

## 6) Reproducing the analysis (step-by-step)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the analysis (from repository root):

```bash
python run.py --data biomet_data.csv --out results
```

3. Generated outputs:
- `results/summary.json` — summary with EER and decidability index per metric and plot paths.
- `results/<metric>/genuine_scores.txt` — 500 genuine scores
- `results/<metric>/imposter_scores.txt` — 49,500 imposter scores
- `results/<metric>/distributions_<metric>.png` — histogram overlay of genuine and imposter scores
- `results/<metric>/far_frr_<metric>.png` — FAR and FRR plotted vs threshold
- `results/<metric>/roc_<metric>.png` — ROC curve (TPR vs FPR)

## 7) Example interpretation from a run on the provided file

After running, you should inspect `results/summary.json`. For the dataset included with this repository I observed the following (these are example results produced by the code you ran):

- Euclidean: EER ≈ 0.4177, Decidability ≈ 0.3219
- Cosine: EER ≈ 0.4771, Decidability ≈ 0.1153

These results indicate that Euclidean matching performs better on this dataset (lower EER, higher d'). The code also saves histograms and ROC plots in `results/` that you can inspect visually — better separability is visible when genuine and imposter distributions overlap less.

## 8) Notes on correctness and potential pitfalls

- Score sign for Euclidean: distances are negated so that higher scores mean better matches. This affects threshold direction: a larger (less negative) value indicates a better match.
- If your data file has a different ordering or if the feature vectors include NaNs, the loader will error; you can pre-process the file or ask me to add robust parsing.
- EER computed here is approximate: the code samples 1000 thresholds uniformly between the global score min/max. For a more precise EER, you can refine the threshold around the crossing point with a binary search.

## 9) Extensions you can request (optional)

- Add AUC (area under ROC) numeric reporting.
- Compute DET curves (log-scaled FAR axis) and plot.
- Compute confidence intervals for EER via bootstrapping.
- Replace template averaging with more robust enrollment (median, PCA-projected templates, or per-user covariance models).
- Add a small web viewer or Jupyter notebook for interactive exploration.

## 10) Files added by me

- `run.py` — main runner
- `src/biometrics.py` — core functions: `load_data`, `make_templates`, `compute_scores`, `compute_far_frr`, `compute_eer`, `decidability_index`
- `src/plotting.py` — plotting helpers
- `requirements.txt` — numpy, scipy, matplotlib
- `README.md` — this file (detailed)

If you want, I can now:

- Add AUC and numeric ROC/AUC values to `results/summary.json` and the README.
- Improve EER precision by refining the threshold search.
- Add a small Jupyter notebook that shows the plots inline and allows interactive threshold selection.

Tell me which of these you'd like next, or paste any runtime/log output you want me to inspect.
 
