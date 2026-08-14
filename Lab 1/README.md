# User Authentication Using Biometric Features

This submission evaluates a biometric authentication system using the supplied feature dataset. The assignment asks for genuine/impostor score distributions, FAR, FRR, ROC, Equal Error Rate, and Decidability Index using both Euclidean distance and cosine similarity.

## Dataset Interpretation

- The file `data/biomet_data.csv` contains `144 x 1000` numeric values.
- The assignment states that there are 100 users, 10 samples per user, and 144 features per sample.
- Therefore, the script transposes the file to `1000 x 144`, then reshapes it as:
  - `100 users x 10 samples x 144 features`
- For each user:
  - Samples 1-5 are used for enrollment/training.
  - Samples 6-10 are used for testing.

## Implementation Summary

The implementation is in `src/biometric_eval.py`.

For each user, the five training feature vectors are averaged to create one enrollment template. Each test sample is compared against all 100 user templates.

For every test sample:

- The comparison with its own user's template is counted as a genuine score.
- The comparisons with the other 99 templates are counted as impostor scores.

This produces:

- `500` genuine scores: `100 users x 5 test samples`
- `49,500` impostor scores: `500 test samples x 99 wrong templates`

Two matching criteria are evaluated:

1. **Euclidean distance** - lower score means a better match.
2. **Cosine similarity** - higher score means a better match.

FAR, FRR, ROC, and EER are computed by sweeping thresholds over the observed score range.

## Results

| Matching criterion | Genuine mean | Genuine std | Impostor mean | Impostor std | EER | EER threshold | Decidability index |
|---|---:|---:|---:|---:|---:|---:|---:|
| Euclidean distance | 291.6160 | 135.5954 | 703.8228 | 301.7948 | 11.50% | 429.8626 | 1.7619 |
| Cosine similarity | 0.9895 | 0.0110 | 0.9604 | 0.0184 | 8.42% | 0.9790 | 1.9254 |

## Performance Curves

### Euclidean Distance

- Genuine/impostor distribution: `results/plots/euclidean_distributions.svg`
- FAR/FRR plot: `results/plots/euclidean_far_frr.svg`
- ROC plot: `results/plots/euclidean_roc.svg`

### Cosine Similarity

- Genuine/impostor distribution: `results/plots/cosine_distributions.svg`
- FAR/FRR plot: `results/plots/cosine_far_frr.svg`
- ROC plot: `results/plots/cosine_roc.svg`

## Conclusion

Cosine similarity performs better on this dataset. It has a lower Equal Error Rate (`8.42%`) than Euclidean distance (`11.50%`) and a higher Decidability Index (`1.9254` vs `1.7619`). This means cosine similarity separates genuine and impostor comparisons more effectively for the supplied biometric feature vectors.

A likely reason is that cosine similarity focuses on the direction of the 144-dimensional feature vector rather than its absolute magnitude. If illumination, acquisition conditions, or sensor changes alter feature magnitudes between enrollment and later testing, cosine similarity can be more stable than raw Euclidean distance.

## How to Run

From the repository root:

```bash
python3 src/biometric_eval.py
```

The script requires NumPy and writes:

- `results/metrics_summary.csv`
- `results/metrics_summary.json`
- SVG plots under `results/plots/`

