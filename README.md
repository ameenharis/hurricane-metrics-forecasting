# Forecasting Hurricane-Relevant Weather Metrics with a Multi-Output MLP

Predicting four hurricane-relevant weather metrics — **rainfall, wind speed, wind direction and surface air pressure** — seven days ahead, from a 1.4-million-row NASA reanalysis dataset.

A single multi-output neural network produces all four forecasts at once, rather than training four separate models.

---

## The problem

Given today's atmospheric conditions at a given latitude and longitude, forecast what those conditions will be a week from now. Four targets, one model.

The complication is that one of the targets is **circular**. Wind direction is measured in degrees, where 359° and 1° are two degrees apart, not 358. A model trained naively on raw degrees learns that those values are at opposite ends of the scale, which is wrong.

## Approach

**Data preparation.** Six atmospheric variables, each supplied as daily CSVs with readings every three hours. Each day's eight readings are averaged to a single daily value, reducing dimensionality while keeping the daily signal. The six variables are merged on location and date, giving 1,404,783 rows.

Targets are constructed by shifting the date forward seven days and joining back on latitude, longitude and date, so each row pairs today's conditions with the conditions one week later. Rows without a matching future observation are dropped, leaving **915,133 usable rows**.

**Circular encoding.** Wind direction is converted to a sine/cosine pair before training and recovered afterwards with `arctan2`, so the model never sees the artificial discontinuity at 0°/360°. This means five model outputs for four physical targets.

**Model.** A three-layer MLP in PyTorch:

```
12 inputs → 128 (ReLU, dropout 0.3) → 64 (ReLU, dropout 0.3) → 5 outputs
```

Adam optimiser, learning rate 0.001, batch size 64, MSE loss, up to 100 epochs with early stopping on validation loss. Features and targets are standardised, with the scaler fitted on the training split only.

**Splits.** 70% training (640,593), 15% validation (137,270), 15% test (137,270).

## Results

Root mean squared error on the held-out test set, in the original units:

| Target | RMSE | Units |
|---|---|---|
| Rainfall | 0.0001 | kg m⁻² s⁻¹ |
| Wind speed | 1.59 | m s⁻¹ |
| Surface pressure | 588.57 | Pa |
| Wind direction | 163.03 | degrees |

Learning curves and predicted-vs-actual plots for each target are in the notebook.

### A note on the wind direction figure

The 163° RMSE is misleading, and worth explaining rather than hiding.

It is computed as a straight-line error on raw degrees, which reintroduces exactly the circularity problem the sine/cosine encoding was designed to avoid. A prediction of 359° against an actual value of 1° is recorded as a 358° error when the true angular distance is 2°. The metric therefore penalises the model heavily for predictions that are, in fact, close.

A fairer measure would be the mean angular error, `min(|θ_pred − θ_true|, 360 − |θ_pred − θ_true|)`. Recomputing on that basis is the first thing I would change here.

Wind direction is also genuinely the hardest of the four targets — it is the least autocorrelated over a seven-day horizon — so some of the error is real. But the reported figure overstates it.

## Running it

Requires Python 3 with `torch`, `pandas`, `numpy`, `scikit-learn` and `matplotlib`.

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

Place the dataset in `data/hurricane/`, with one sub-folder per variable (`Rainf_tavg`, `Wind_f_inst`, `Wind_dir_p1000`, `Psurf_f_inst`, `SWdown_f_tavg`, `LWdown_f_tavg`), or point the notebook elsewhere:

```bash
export DATA_DIR=/path/to/dataset
```

Then run `hurricane_mlp.ipynb` top to bottom. The full data load takes a few minutes; training runs on CPU.

## What I would do differently

- **Fix the wind direction metric** to measure angular distance, as above.
- **Evaluate per-target rather than in aggregate.** With one model producing five correlated outputs, a single averaged loss can hide poor performance on an individual target — which is close to what happened here.
- **Keep some temporal structure.** The train/test split is random across all rows, so observations from adjacent days can land on both sides of the split. For a forecasting problem a chronological split would be a more honest test of generalisation.
- **Try per-target loss weighting.** The four targets differ by orders of magnitude in scale; standardisation handles this during training, but the model still has to trade them off against one another.

## Files

- `hurricane_mlp.ipynb` — full pipeline: loading, feature engineering, training, evaluation, plots
- `predictions.csv` — test-set predictions alongside actuals for all four targets
