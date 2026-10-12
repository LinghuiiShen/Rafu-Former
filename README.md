# Rafu-Former

**COMP4434 — Big Data Analytics Course Project (2025)**

A time-series forecasting project on the Electricity dataset. Rafu-Former combines cross-variable attention and dynamic basis decomposition for univariate and multivariate forecasting, with GRU, LSTM, and TCN baselines and experiments on pretraining and attention ablation.

**Technologies:** Python, PyTorch, NumPy, pandas, scikit-learn, Matplotlib, and TensorBoard.

## Repository Structure

- `main.py`, `pretrain.py`: forecasting and pretraining entry points.
- `model.py`, `layers/`: Rafu-Former architecture.
- `data_provider/`: dataset loading and time features.
- `gru.py`, `lstm.py`, `tcn_2.py`: baseline models.
- `script/`: experiment scripts.
- `docs/`: [project report](docs/Rafu-Former%20Report.pdf) and [original usage instructions](docs/original-readme.txt).

## Getting Started

Install dependencies with `pip install -r requirements.txt`, and place the Electricity dataset at `datasets/electricity/electricity.csv`. Run experiments from the repository root:

```bash
bash script/M.sh     # Multivariate forecasting
bash script/S.sh     # Univariate forecasting
bash script/Mpre.sh  # Pretraining followed by forecasting
```

The scripts contain machine-specific GPU indices in `CUDA_VISIBLE_DEVICES`; adjust them to match your environment. The dataset is not included.

## Acknowledgements

This course project reuses parts of BasisFormer, including data-loading and evaluation utilities, as documented in the original project instructions.
