# Biometric User Authentication — Lab 1

This submission evaluates a biometric user-authentication system using the provided feature dataset. The analysis computes genuine and impostor score distributions, FAR/FRR curves, ROC, Equal Error Rate (EER), and the Decidability Index for two matching criteria: Euclidean distance and cosine similarity.

## Dataset and Preprocessing

- Source file: `data/biomet_data.csv` containing 144 × 1000 numeric values.
- Problem interpretation: 100 users × 10 samples per user × 144 features per sample.
- Preprocessing: the CSV is transposed to 1000 × 144 and reshaped to `100 users × 10 samples × 144 features`.
- Protocol: for each user, samples 1–5 are used for enrollment (training) and samples 6–10 for testing.

### Practical notes and example (Python)

To reproduce the preprocessing steps used by `src/biometric_eval.py`:

```python
import numpy as np

# load raw CSV with shape (144, 1000)
raw = np.loadtxt('data/biomet_data.csv', delimiter=',')

# transpose to (1000, 144) and reshape to (100, 10, 144)
data = raw.T.reshape((100, 10, 144))

# enrollment samples indices: 0..4; test indices: 5..9
enroll_idx = range(0, 5)
test_idx = range(5, 10)
```

## Implementation Overview

- Code: `src/biometric_eval.py` implements the full evaluation pipeline.
- Enrollment: for each user the five enrollment feature vectors are averaged to produce a single template.
- Matching: every test sample is compared against all 100 user templates. A comparison with the sample's own user template yields a genuine score; comparisons with the other 99 templates yield impostor scores.

This yields:

- 500 genuine scores (100 users × 5 test samples)
- 49,500 impostor scores (500 test samples × 99 non-matching templates)

Two matching criteria are evaluated:

1. Euclidean distance — lower values indicate better matches.
2. Cosine similarity — higher values indicate better matches.

FAR, FRR, ROC and EER are obtained by sweeping decision thresholds across the empirical score ranges for each criterion.

### Matching formulas

- Euclidean distance between vectors $x$ and $y$:

  $$d(x,y)=\lVert x-y \rVert_2=\sqrt{\sum_{i=1}^{144}(x_i-y_i)^2}\;.$$ 

- Cosine similarity:

  $$s_{cos}(x,y)=\frac{x\cdot y}{\lVert x\rVert_2\,\lVert y\rVert_2}\;,$$

For Euclidean distance a smaller $d$ indicates a better match; for cosine similarity a larger $s_{cos}$ indicates a better match.

## Key Results

| Matching criterion | Genuine mean | Genuine std | Impostor mean | Impostor std | EER | EER threshold | Decidability index |
|---|---:|---:|---:|---:|---:|---:|---:|
| Euclidean distance | 291.6160 | 135.5954 | 703.8228 | 301.7948 | 11.50% | 429.8626 | 1.7619 |
| Cosine similarity | 0.9895 | 0.0110 | 0.9604 | 0.0184 | 8.42% | 0.9790 | 1.9254 |

## Performance Visualizations

All plots are written to the `results/plots/` directory.

- Euclidean distance:
  - Distributions: `results/plots/euclidean_distributions.svg`
  - FAR/FRR: `results/plots/euclidean_far_frr.svg`
  - ROC: `results/plots/euclidean_roc.svg`

- Cosine similarity:
  - Distributions: `results/plots/cosine_distributions.svg`
  - FAR/FRR: `results/plots/cosine_far_frr.svg`
  - ROC: `results/plots/cosine_roc.svg`

## Conclusion

On this dataset cosine similarity provides better separation between genuine and impostor comparisons: it attains a lower EER (8.42%) and a higher Decidability Index (1.9254) than Euclidean distance (EER = 11.50%, Decidability = 1.7619). This improvement likely stems from cosine similarity's emphasis on vector direction rather than magnitude, making it more robust to scaling effects introduced by acquisition or sensor variability.

## Evaluation Metrics — definitions and interpretation

- False Accept Rate (FAR): fraction of impostor comparisons that are incorrectly accepted at a given threshold. If `N_imp` is the number of impostor trials and `FP` is false accepts, then

  $$\mathrm{FAR}=\frac{\mathrm{FP}}{N_{imp}}\;.$$ 

- False Reject Rate (FRR): fraction of genuine comparisons that are incorrectly rejected. If `N_gen` is the number of genuine trials and `FN` is false rejects, then

  $$\mathrm{FRR}=\frac{\mathrm{FN}}{N_{gen}}\;.$$ 

- Equal Error Rate (EER): the operating point where FAR and FRR are equal. Lower EER indicates better overall discrimination.

- Decidability Index (DI): a measure of separation between genuine and impostor score distributions. For genuine distribution mean $\mu_g$ and std $\sigma_g$, and impostor mean $\mu_i$ and std $\sigma_i$, the DI used is:

  $$\mathrm{DI}=\frac{|\mu_g-\mu_i|}{\sqrt{\tfrac{1}{2}(\sigma_g^2+\sigma_i^2)}}\;.$$ 

When explaining results to others, emphasize: the means/stds show central tendency and spread; DI normalizes the separation by combined variance; EER summarizes operating performance with a single number.

## Script structure and how to explain it

`src/biometric_eval.py` is organized into the following conceptual steps (look for the similarly named functions inside the file):

- `load_data()` — loads CSV and reshapes to `(100, 10, 144)`.
- `build_templates(enroll_idx)` — averages enrollment samples per user to produce templates of shape `(100, 144)`.
- `compute_scores(templates, test_samples)` — computes pairwise comparisons between each test sample and all templates, returning genuine and impostor score arrays for each criterion.
- `compute_metrics(scores)` — sweeps thresholds to compute FAR/FRR curves, ROC and finds EER and the EER threshold.
- `plot_results(...)` — draws distribution histograms, FAR/FRR curves and ROC curves and saves SVGs to `results/plots/`.
- `save_results()` — writes `results/metrics_summary.csv` and `results/metrics_summary.json`.

Point out these steps when presenting: data → templates → scores → metrics → plots. Each step transforms data into a more interpretable representation.

## Outputs and what they mean

- `results/metrics_summary.csv` — tabular metrics (means, stds, EER, DI) per matching criterion.
- `results/metrics_summary.json` — same metrics in machine-readable form.
- `results/plots/*.svg` — visualizations:
  - `*_distributions.svg` shows overlapping genuine and impostor histograms/density estimates.
  - `*_far_frr.svg` shows FAR and FRR vs threshold (useful to point out operating points).
  - `*_roc.svg` shows ROC curve and AUC (if computed).

## How to run (reproducible steps)

1. Create a Python environment and install NumPy (and Matplotlib if plotting is required). Example using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

2. Run the evaluation from the repository root:

```bash
python3 src/biometric_eval.py
```

3. Review results in `results/` and open the SVGs from `results/plots/`.

If you want, I can add a `requirements.txt` and a small wrapper to change the enrollment/test split from the command line.

## Tips for explaining this lab to your assistant

- Start by describing the dataset shape and the enrollment/testing protocol — these determine the number of genuine and impostor trials.
- Walk through the processing pipeline (template creation, scoring, threshold sweep).
- Explain each metric briefly (FAR, FRR, EER, DI) and why EER and DI are useful summary numbers.
- Use the distribution plots to visually demonstrate overlap vs separation — point to the EER threshold on the FAR/FRR plot.
- Mention why cosine similarity might outperform Euclidean distance here (direction vs magnitude, robustness to scaling).

---

If you'd like, I can:

- add `requirements.txt` and example `pip` commands,
- add a short script that prints intermediate counts (number of genuine/impostor scores) for demonstration, or
- annotate `src/biometric_eval.py` with docstrings and inline comments to make it presentation-ready.

## Requirements and Usage

- Requirements: Python 3 and NumPy.
- From the repository root run:

```bash
python3 src/biometric_eval.py
```

The script produces:

- `results/metrics_summary.csv`
- `results/metrics_summary.json`
- SVG plots in `results/plots/`

If you would like, I can also add a `requirements.txt` and a brief usage example showing how to change the enrollment/testing split or evaluate only one matching criterion.

