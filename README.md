# PANDA

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Reproduce the figure using the released checkpoint:

```bash
python evaluate_dimension_generalization.py
python plot_dimension_generalization.py
```

Quick test with fewer tasks:

```bash
python evaluate_dimension_generalization.py --num-seeds 2 --tasks-per-seed 4
python plot_dimension_generalization.py
```

Train from scratch and regenerate:

```bash
python train.py
python evaluate_dimension_generalization.py
python plot_dimension_generalization.py
```

## Project structure

```
.
├── data.py
├── evaluate_dimension_generalization.py
├── panda_model.py
├── plot_dimension_generalization.py
├── README.md
├── requirements.txt
├── train.py
├── checkpoints/
│   └── panda_dimension_generalization.pt
└── results/
    ├── dimension_generalization_scm_panda.json
    └── dimension_generalization_scm_panda.png
```
