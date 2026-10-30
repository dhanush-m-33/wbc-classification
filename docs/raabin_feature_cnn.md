# RAABIN feature CNN

This standalone CNN uses only the RAABIN `train` manifests and their processed
images. Its ten training targets are derived from the WBC class, not from
human annotations of each morphology feature. Treat its output as a class-based
estimate, not verified visual feature recognition. `dense_blue_granules` has
no positive target in the current class-to-feature mapping.

## Train

From the repository root, install project dependencies if needed, then run:

```powershell
python -m pip install -r requirements.txt
python src\train_raabin_feature_cnn.py train
```

The script joins `data\manifests\raabin_train.csv` to
`data\manifests\processed_raabin_train.csv`, uses existing processed PNGs, and
creates a group-separated validation split. It does not read TestA or TestB.
You can change the epoch cap or seed:

```powershell
python src\train_raabin_feature_cnn.py train --epochs 25 --seed 42
```

## Outputs

- `models\raabin_feature_cnn.keras` — best saved model.
- `results\raabin_feature_cnn_validation_with_vectors.csv` — held-out validation predictions,
  weak targets, image/class metadata, per-feature probabilities, and a
  `feature_vector` JSON array ordered by `feature_names` in
  `features\raabin_morphology_feature_mapping.json`.
- `reports\raabin_feature_cnn_metrics.json` — validation metrics and limitations.
- `reports\raabin_feature_cnn_history.csv` — per-epoch training history.

The validation CSV is the result to inspect first. Its metrics measure agreement
with class-derived weak targets, not human-verified morphology.

## Predict on processed images

Pass one or more processed PNG paths to load the saved model and write a
prediction CSV:

```powershell
python src\train_raabin_feature_cnn.py predict --images data\processed\raabin\train\neutrophil\some_cell.png --output results\my_feature_predictions.csv
```

Each feature gets a probability and a binary prediction using a `0.5` cutoff.
The CSV's `feature_vector` column contains the ten continuous probabilities in
the mapping's `feature_names` order. Use processed PNGs in the same format as
the RAABIN training images.
