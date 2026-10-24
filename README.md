# Paired oasis-boundary analysis

This repository contains the core non-graphical analysis used to train and evaluate the paired boundary ranking (PBR) model. Each endpoint is represented by an ordered 18-variable vector: five prior probabilities and 13 Harmonized Landsat and Sentinel-2 features. Missing values are handled inside the fitted pipeline.

## Repository contents

- `src/oasis_pair_analysis.py`: feature schema, model pipeline, stable-key checks, endpoint metrics, strict-pair metrics, and Wilson intervals.
- `scripts/train_model.py`: fits and serializes the endpoint classifier.
- `scripts/evaluate_model.py`: evaluates a fitted model without refitting.
- `requirements.txt`: versions used for the validated implementation.

## Dependencies

Python 3.13.9 was used with NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, scikit-learn 1.7.2, and joblib 1.5.2.

## Run

The training and evaluation tables must each contain `pair_id`, `endpoint_id`, `endpoint_role`, `expected_label`, and the 18 feature columns defined in `src/oasis_pair_analysis.py`. Each pair must contain exactly two unique endpoints, one with expected label 0 and one with expected label 1. Pair identities are joined by key rather than row position. Samples used for independent evaluation must not occur in the training table, and both endpoints from a pair must remain in the same data split.

```bash
python scripts/train_model.py --input training_endpoints.csv --model-out model.joblib --audit-out training_audit.json
python scripts/evaluate_model.py --input independent_evaluation_endpoints.csv --model model.joblib --summary-out evaluation_summary.json --pairs-out pair_results.csv
```

The classifier is `SimpleImputer(strategy="median", add_indicator=True) -> StandardScaler -> LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=9801, class_weight="balanced")`. Endpoint predictions use a probability threshold of 0.5. Pair success requires the expected oasis endpoint to have a strictly larger probability than the paired non-oasis endpoint; ties fail. Wilson intervals use `z=1.96`.

The five precomputed prior columns have the following scientific roles: `d531_probability` is a vegetation-water index composite from NDVI, NDMI, and NDWI; `d532_hls_probability` combines the five HLS index summaries with the valid-observation count; `d532_probability` represents NIR-red and SWIR1-SWIR2 reflectance contrasts; `d533_probability` combines the NDVI range with visible-NIR and SWIR observation support; and `d534_selected_probability` is the unique maximum of those four prior scores, with a tie within `1e-12` represented as missing. The code in this repository begins from the precomputed ordered 18-feature endpoint table.

## Data

HLS v2.0 L30/S30 surface-reflectance inputs and the public geospatial products used by the study are available from their original providers. The endpoint-level analysis table is available from the corresponding author upon reasonable request; correspondence details will be provided with the associated article.
