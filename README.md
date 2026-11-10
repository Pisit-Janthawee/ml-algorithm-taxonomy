# ML Algorithm Taxonomy

A visual, illustrated map of machine-learning algorithms and the statistics behind them, drawn as a multi-page
[draw.io](https://www.drawio.com/) diagram.

Open **`outputs/ML Algorithm Taxonomy.drawio`** in draw.io (desktop, web, or the VS Code draw.io extension).

## What's inside

- **Overview page:** the full taxonomy tree (supervised, unsupervised, deep learning & NLP, reinforcement learning,
  time series, recommender systems, foundations, ML workflow), with the ★ base (root) algorithm of each category highlighted.
  Click a category to open its page.
- **One page per category (27):** each starts from its base algorithm (e.g. Simple Linear Regression, ID3, Bagging,
  K-Means, PCA, MLP, Q-Learning, ARMA, CLT, ADF, Triple-Barrier labelling) and shows:
  - the "built on" chain of background algorithms
  - worked examples with real computed numbers (tables, plots, formulas in LaTeX)
  - data illustrations and hyperparameter tuning
  - ⚠ assumptions (how to check them, what to do if they are violated)
  - 💡 hidden / versatile uses
  - ❓ "Doing this answers…" callouts that explain how to read each component

## Structure

| Path | Contents |
|---|---|
| `inputs/ML Empiral Diagram.drawio` | Original hand-made diagram (style reference) |
| `outputs/ML Algorithm Taxonomy.drawio` | Generated taxonomy diagram |
| `scripts/` | Python generator: see [`scripts/README.md`](scripts/README.md) |

## Rebuild

```bash
cd scripts
python -I build.py "../outputs/ML Algorithm Taxonomy.drawio"
```

Pure Python 3 (standard library only). `render.py` (optional, for PNG previews) needs Google Chrome.
Rebuilding overwrites manual edits made in draw.io.
