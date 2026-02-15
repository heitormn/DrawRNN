# DrawRNN — Quick, Draw! Sketch Classifier

An LSTM-based Recurrent Neural Network that classifies hand-drawn sketches from Google's [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.

The model reads variable-length stroke sequences (x, y, stroke progress) and predicts which category a drawing belongs to.

---

## Project Structure

```
DrawRNN/
├── main.py                 # CLI entry point (train & evaluate)
├── requirements.txt        # Python dependencies
├── .gitignore
├── LICENSE
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters & path defaults
│   ├── dataset.py          # Data loading, processing, Dataset & DataLoader
│   ├── model.py            # DrawRNNModel (multi-layer LSTM)
│   ├── train.py            # Training loop with early stopping
│   └── visualize.py        # Sketch rendering & training curves
│
├── data/
│   └── Drawings/           # Place .ndjson files here
│
├── checkpoints/            # Saved model weights
│
└── notebooks/
    └── DrawRNN.ipynb       # Interactive exploration notebook
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DrawRNN.git
cd DrawRNN
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the data

Download the **Simplified Drawing** files (`.ndjson`) from the
[Quick, Draw! dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified)
and place them in `data/Drawings/`.

The files should be named `full_simplified_<category>.ndjson`, e.g.:

```
data/Drawings/full_simplified_ant.ndjson
data/Drawings/full_simplified_axe.ndjson
data/Drawings/full_simplified_bat.ndjson
```

> **Tip:** You can use `gsutil` to download:
> ```bash
> gsutil cp gs://quickdraw_dataset/full/simplified/ant.ndjson data/Drawings/full_simplified_ant.ndjson
> ```

---

## Usage

### Train (default settings)

```bash
python main.py
```

### Train with custom hyperparameters

```bash
python main.py --epochs 100 --lr 0.0005 --hidden 256 --layers 3 --batch-size 128
```

### Evaluate a saved checkpoint

```bash
python main.py --test checkpoints/best_model.pt
```

### Add more categories

```bash
python main.py --categories ant axe bat cat dog
```

> Make sure the corresponding `.ndjson` files are in `data/Drawings/`.

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/Drawings` | Path to ndjson data folder |
| `--categories` | `ant axe bat` | Categories to train on |
| `--hidden` | `128` | LSTM hidden size |
| `--layers` | `4` | Number of LSTM layers |
| `--dropout` | `0.0` | Dropout between LSTM layers |
| `--epochs` | `50` | Maximum training epochs |
| `--lr` | `0.001` | Learning rate (AdamW) |
| `--batch-size` | `256` | Mini-batch size |
| `--patience` | `5` | Early stopping patience |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `--test` | — | Path to checkpoint (eval-only) |

---

## Model Architecture

```
DrawRNNModel(
  (lstm): LSTM(3, 128, num_layers=4, batch_first=True)
  (fc):   Linear(128, num_classes)
)
```

- **Input**: Variable-length sequence of `(x, y, stroke_progress)` per drawing
- **Encoding**: 4-layer LSTM with 128 hidden units
- **Output**: Softmax over category classes (via CrossEntropyLoss)
- **Optimizer**: AdamW with early stopping

---

## Data Format

Each drawing from Quick, Draw! is a JSON object:

```json
{
  "word": "ant",
  "drawing": [
    [[x1, x2, ...], [y1, y2, ...]],
    [[x1, x2, ...], [y1, y2, ...]]
  ]
}
```

The preprocessing converts strokes into a flat sequence of `(x, y, stroke_progress)` tensors, where `stroke_progress` linearly interpolates from 0 → 1 within each stroke.

---

## Results

With default settings (3 classes, 128 hidden, 4 LSTM layers, 50 epochs max):

| Metric | Value |
|--------|-------|
| Test Accuracy | ~95%+ |
| Parameters | ~400K |


---

## Acknowledgments

- [Google Quick, Draw!](https://quickdraw.withgoogle.com/data) for the dataset
- Inspired by Chapter 13 (RNNs) of *Hands-On Machine Learning* by Aurélien Géron
