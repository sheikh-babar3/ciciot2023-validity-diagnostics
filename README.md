# Closed-form validity diagnostics for class-imbalanced IoT IDS benchmarks

Code and result files for the paper *"Closed-form validity diagnostics for class-imbalanced IoT intrusion-detection benchmarks: detecting non-classifying models from published metrics alone"* (Hameed, Hayat, Rizwan, Bagdasar, Kadar).

## Contents
| Path | What it is |
|---|---|
| `notebooks/v10_FULL_RUN_as_executed.ipynb` | The evaluation notebook exactly as run to produce the paper's results (5-fold CV, 20 methods, 4,258,329 rows). |
| `notebooks/v10_FULL_RUN_patched_A1_stability.ipynb` | Same pipeline with a numerically stabilised Transformer baseline (A1) and NaN-safe ROC plotting. **Not used for the paper's numbers.** |
| `results/per_fold_results.csv` | Accuracy, macro-P/R/F1, AUC and time per method and fold. |
| `results/per_class_metrics.csv` | Per-class precision/recall/F1 per method and fold. |
| `results/table_III_summary.csv`, `avg_ranks.csv`, `nemenyi_pvalues.csv` | Fold means, Friedman average ranks, Nemenyi p-values. |
| `results/cascade_stage_resolution.csv` | Share of each test fold resolved at each cascade stage. |
| `results/loss_history/` | Saved training-loss histories (`.npy`). |
| `results/figures/` | Figures produced by the notebook. |
| `code/` | Scripts that regenerate the paper's figures from `results/` (run from the repository root; paths are relative). |

## Data
The raw data are the first eighteen part files (`part-00000` ... `part-00017`) of the public CICIoT-2023 release (Neto et al., *Sensors* 23(13):5941, 2023): https://www.unb.ca/cic/datasets/iotdataset-2023.html . They are not redistributed here. Download them and set the data path in the first cells of the notebook.

## Reproducing
1. Create an environment (Python, PyTorch, scikit-learn, imbalanced-learn, pandas, numpy, scipy, matplotlib; see `requirements.txt`).
2. Download the part files and point the notebook at them.
3. Run the notebook top to bottom. Intermediate results are checkpointed per fold and method.
4. `python code/srfigs.py` regenerates the supporting figures from `results/` (edit the `R`/`O` paths at the top if needed).

Hardware used: NVIDIA RTX 4060 (8 GB), Intel Core i7-12700, 32 GB RAM, WSL2 Ubuntu.

## Citing
See `CITATION.cff`. Please cite the paper and the archived release (Zenodo DOI to be added).

## License
Code: MIT (see `LICENSE`). Result files: CC BY 4.0. The CICIoT-2023 dataset has its own terms from the Canadian Institute for Cybersecurity.
