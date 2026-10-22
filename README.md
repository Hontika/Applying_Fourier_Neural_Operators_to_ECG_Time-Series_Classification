# Fourier neural operators for ECG time-series classification

Code for the ECG classification study. Selected trained checkpoints are distributed separately in the [model checkpoint folder on Google Drive](https://drive.google.com/drive/folders/1Lx4NkrypAOVdMpiZumG1pDW8qYmMPM5L?usp=sharing). `Pipeline.ipynb` contains data preparation, model definitions, training, and evaluation. The experiment treats ECG5000 as its primary task and trains separately on three supporting datasets; checkpoints are not transferred between datasets.

## Datasets

| Dataset | Source | Study task | Study split |
| --- | --- | --- | --- |
| [ECG5000](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) | UCR/UEA Time Series Classification Archive | 5,000 univariate, 140-point heartbeats; 5 classes | Original archive files pooled, then stratified 3,000/1,000/1,000 train/validation/test |
| [NonInvasiveFetalECGThorax1](https://www.timeseriesclassification.com/description.php?Dataset=NonInvasiveFetalECGThorax1) | UCR/UEA Time Series Classification Archive | 3,765 univariate, 750-point series; 42 classes | Original archive files pooled, then stratified 2,259/753/753 |
| [NonInvasiveFetalECGThorax2](https://www.timeseriesclassification.com/description.php?Dataset=NonInvasiveFetalECGThorax2) | UCR/UEA Time Series Classification Archive | 3,765 univariate, 750-point series; 42 classes | Original archive files pooled, then stratified 2,259/753/753 |
| [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | PhysioNet | 21,388 usable 10-second Lead II recordings at 100 Hz; binary NORM/ABNORMAL | Patient-respecting folds 1-8/9/10: 17,084/2,146/2,158 |

The PTB-XL counts are **after** excluding records without a diagnostic superclass. A record is NORM only when normal is its sole diagnostic superclass; other retained records are ABNORMAL. For the three UCR/UEA datasets, these are the study's new splits, **not** the archives' original train/test divisions. Download the data from the linked providers and observe their terms; raw datasets are not included here.

## Selected checkpoints

The [separate checkpoint package](https://drive.google.com/drive/folders/1Lx4NkrypAOVdMpiZumG1pDW8qYmMPM5L?usp=sharing) contains the five highest **validation macro-F1** runs from each of 13 model families on each of the four datasets, in the `e50` training regime: **260 checkpoints** in total. Held-out test scores are reported after selection and do not determine membership or ranking. These are the runs used for the manuscript's top-five candidate distributions, not the full search and not necessarily every seed-aggregated or parameter-matched comparison.

Each run is at `<dataset>/<family>/e50/<run_id>/` within the package and contains `model.pt` (PyTorch `state_dict`), `config.json`, and `metrics.json`. The package's `manifest.csv` records the family, rank, seed, validation and test macro-F1, parameter count, relative path, and checkpoint checksum for every run. Reconstruct a model using the matching family and configuration in `Pipeline.ipynb` before loading its `state_dict`.

The checkpoint package is not part of this Git repository, so cloning the code alone will not provide the models. Install Python dependencies from `Requirements.txt` in a compatible environment; the pinned PyTorch packages use the CUDA 12.8 wheel index.

The full experiment outputs, datasets, and manuscript source are deliberately not tracked in this repository. The selected checkpoint package is a portable subset of the completed experiment.
