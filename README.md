# Cassava Leaf Disease Classification

Streamlit demo comparing EfficientNet-B0, ResNet18, and MobileNetV3-Large on four cassava leaf classes.
Macro-F1 is the primary evaluation metric. The trained checkpoints use `cbs`
for Brown Leaf Spot (called `bls` in the project plan).

## Run on CPU

Use Python 3.12 and a separate virtual environment for deployment:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The Windows GPU training environment remains separate and uses the cu128 index.
Raw datasets and training environments are excluded from Git.
Inference checkpoints contain the original model weights without optimizer state.

## Streamlit Community Cloud

- Branch: `main`
- Entrypoint: `app.py`
- Python: `3.12`
- Dependencies: root `requirements.txt`

Evaluation CSVs, charts, and all three inference checkpoints are included in the repository.
