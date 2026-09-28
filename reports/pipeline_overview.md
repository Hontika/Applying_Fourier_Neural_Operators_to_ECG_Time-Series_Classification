# Experimental Pipeline Overview

## Introduction

The aim of the study was to investigate whether Fourier Neural Operator (FNO) improve the classification of physiological time series, namely ECG datasets. The main one in this study was ECG5000. We asked whether the datasets contain spectral structure that is meaningful for classification, and whether modeling this structure can improve performance beyond standard convolutional or residual architectures.

For several baseline model families, a corresponding version enhanced with an FNO layer was implemented so that the effect of spectral modeling could be studied under similar training and evaluation conditions.

We tried to evaluate whether FNO-based spectral operators provide a measurable classification benefit. The challenge was to generate reproducible results for comparing FNO-enhanced models against the baselines.

## Datasets and Input Format

The main dataset in the experiment was `ECG5000`. Each sample was treated as a univariate sequence of length `L`, and class labels are mapped from their original string representation to integer indices in the range `0, ..., K-1`.

The data loader performs the following steps:

- parse the `.ts` files after the `@data` marker;
- split each record into a comma-separated time series and its label;
- convert the time series to `float32`;
- build a consistent label-to-index mapping from the training labels;
- validate that all test labels are represented in the training label set.

## Data Partitioning Strategy

After loading the original train and test files, the current pipeline concatenates them and creates a new stratified split over the pooled dataset. The effective partition is:

- 60% training
- 20% validation
- 20% test

This is performed through a two-stage stratified split with fixed `random_state=42`. As a consequence, the experiments do not preserve the original benchmark train/test split; instead, they use a project-specific stratified repartitioning. For a scientific paper, this point should be stated explicitly because it affects comparability with published benchmark results.

## Preprocessing

### Signal normalization

Each sample is normalized independently by z-score normalization:

\[
\hat{x} = \frac{x - \mu(x)}{\sigma(x)}
\]

where the mean and standard deviation are computed per sample.

### Optional augmentation

For raw 1D inputs, the dataset class supports optional random temporal shifting with zero padding rather than circular wrap-around. The default configuration in the current code keeps this augmentation disabled:

- `AUG_TIME_SHIFT = False`
- `MAX_SHIFT = 5`

### Spectrogram conversion

For image-like baselines, a dedicated dataset class converts each 1D signal into a log-magnitude short-time Fourier transform (STFT) spectrogram. The transformation includes:

- Hann windowing;
- configurable `n_fft` and `hop_length`;
- magnitude extraction;
- logarithmic compression with a small epsilon for numerical stability;
- optional in-memory caching of spectrograms to reduce repeated computation across epochs.

The resulting tensor has shape `(1, F, T)`, where `F` is the frequency dimension and `T` is the time-frame dimension.

## Data Loaders

The standard data loaders use:

- training batch size: `128`
- validation batch size: `256`
- test batch size: `256`

For training, shuffling is enabled and `drop_last=True` is used. Validation and test loaders are deterministic and unshuffled.

## Model Families

The model suite was designed to separate ordinary architectural improvements from the specific contribution of Fourier-based spectral modeling. All models ultimately produce class logits for multi-class classification, but the key comparisons are between standard baselines and variants that insert FNO or Fourier-inspired blocks.

### 1D models operating on raw signals

#### Compact 1D U-Net

The compact 1D U-Net serves as one of the main non-Fourier baselines. It uses a compact encoder-decoder structure with skip connections, 1D convolutions, max-pooling in the encoder, transposed convolutions in the decoder, and an adaptive average pooling classification head.

#### 1D U-Net with FNO bottleneck

This variant is the direct Fourier-enhanced counterpart of the compact U-Net. It replaces the plain bottleneck with Fourier Neural Operator blocks, allowing the model to learn transformations over low-frequency spectral modes of the latent signal representation. The spectral block performs:

- real FFT along the temporal axis;
- truncation to a fixed number of low-frequency modes;
- learned complex-valued mixing in the frequency domain;
- inverse FFT back to the time domain.

This design tests whether class-relevant information is better captured when the model is allowed to manipulate global frequency components rather than relying only on local convolutional filters.

#### Attention U-Net

The attention-based variant is included as a stronger non-Fourier baseline. It applies learned attention gates to the skip connections so that decoder features selectively fuse encoder information.

#### Attention U-Net with FNO bottleneck

This model evaluates whether spectral modeling remains beneficial when the baseline already contains attention-gated skip connections. It combines:

- attention-gated skip fusion;
- an FNO-enhanced bottleneck representation.

#### ResNet1D baseline

The repository includes a residual 1D convolutional baseline roughly analogous to a lightweight ResNet-18-style design adapted for univariate sequences.

#### ResNet-9 style models

Three additional 1D residual variants are defined:

- `ResNet9_1DClassifier`
- `CFResNet9_1DClassifier`
- `FNOResNet9_1DClassifier`

These extend the baseline comparison beyond U-Net-like models and introduce alternative residual architectures, including Fourier-enhanced residual blocks.

### Spectrogram-based models

#### Spectrogram CNN

This model provides a second way of testing the spectral hypothesis. Instead of inserting FNO blocks into raw-signal models, the signal is first transformed into a time-frequency representation using STFT, after which a CNN processes the resulting spectrogram. This helps assess whether spectral information is useful even without a dedicated FNO module.

#### Spectrogram CNN with FNO

The corresponding Fourier-enhanced model augments the 2D convolutional frontend with 2D spectral blocks applied in the spectrogram domain. This tests whether explicit Fourier operators provide additional benefit even after the signal has already been represented in time-frequency form.

## Training Procedure

The training loop is implemented in a modular fashion through:

- `train_one_epoch`
- `eval_model`
- `fit`

The default supervised optimization setup uses:

- loss: cross-entropy loss
- optimizer: `AdamW`
- model selection criterion: best validation accuracy

For fixed-length training regimes, the pipeline stores the weights associated with the best validation accuracy observed during training and restores those weights at the end.

## Training Regimes

The code supports two styles of training schedule.

### Fixed-epoch regimes

These are encoded as strings such as:

- `e50`
- `e120`

The helper `regime_to_epochs` converts such names into integer epoch counts.

### Plateau-based early stopping

The `plateau` regime trains until validation performance stops improving under the following logic:

- minimum number of epochs before stopping is allowed;
- hard upper cap on the number of epochs;
- patience counter based on insufficient improvement;
- improvement monitored jointly through validation accuracy and validation loss.

The current plateau configuration is:

- `min_epochs = 40`
- `max_epochs = 300`
- `patience = 20`
- `min_delta_acc = 1e-4`
- `min_delta_loss = 1e-4`

The best checkpoint is selected by prioritizing validation accuracy, with validation loss used as a tie-breaker.

## Hyperparameter Search

The notebook implements a resumable grid-search backend that enumerates model configurations, skips runs already completed on disk, and stores every run in a structured directory tree. The search is not only for maximizing classification accuracy; it is also intended to make the Fourier and non-Fourier comparisons more reliable by evaluating several capacity, regularization, and spectral-mode settings.

### Shared search space

- `base`: `16`, `32`, `48`
- learning rate: `1e-3`, `5e-4`
- weight decay: `0.0`, `1e-4`
- dropout: `0.0`, `0.1`, `0.2`

### FNO-specific search space

- `modes`: `8`, `16`, `24`
- `fno_layers`: `1`, `2`

### Spectrogram-specific search space

- `n_fft`: `32`, `64`
- `hop_length`: `4`, `8`, `16`

### Spectrogram FNO-specific search space

- `modes_h`: `8`, `16`
- `modes_w`: `4`, `8`
- `fno_layers_2d`: `1`, `2`, `3`

### Residual Fourier variants

- compact Fourier residual branch: `modes = 8, 16, 24`
- FNO ResNet-9 branch: `modes = 8, 16, 24`, `fno_layers = 1, 2`

## Run Management and Persistence

Each run is saved under a directory of the form:

`models/<DATASET_NAME>/<family>/<train_regime>/run_xxx/`

Each run folder stores at least:

- `model.pt` for trained weights;
- `config.json` for the run configuration;
- `metrics.json` for recorded metrics.

The pipeline also maintains a `leaderboard.csv`, although completion detection is based on scanning the filesystem rather than relying exclusively on the leaderboard. This makes the grid search resumable and robust to interrupted sessions.

## Evaluation Protocol

After training, the pipeline scans the saved run directories and reloads each model to evaluate it on the test split. The evaluation protocol is designed to answer whether Fourier-enhanced models consistently outperform their closest non-Fourier counterparts, and whether any observed gains remain meaningful when accounting for model size. The main reported metrics are:

- test accuracy;
- macro-F1 score;
- balanced accuracy;
- parameter count in millions.

The evaluation stage then derives several aggregate result tables, including:

- `eval_stats_all.csv`
- `eval_best_per_family_per_regime.csv`
- `eval_best_overall_per_family.csv`
- `top5_per_family_per_regime.csv`
- `param_matched_cross_family.csv`

This organization supports both standard model ranking and more careful family-wise comparison under approximate parameter-count matching. The parameter-matched comparison is especially important for the study question, because an FNO model should not be considered better simply because it has substantially more parameters.

## Visualization and Reporting Outputs

The notebook produces a range of manuscript-oriented outputs aimed at interpreting the Fourier-versus-non-Fourier comparison, including:

- confusion matrices for selected best models;
- per-class F1 visualizations;
- parameter-count versus macro-F1 scatter plots;
- regime-comparison bar charts and heatmaps;
- top-5 distribution boxplots for each family;
- dataset imbalance summary tables;
- LaTeX-exportable comparison tables.

Figures are exported under `figs/<DATASET_NAME>/`, while table-oriented outputs are written to `tables/` and `dataset_balance_tables/`.

## Interpretability Utilities

The pipeline includes Grad-CAM-style utilities for 1D models. These utilities are intended to highlight temporal regions that contribute most strongly to the predicted class, which may be valuable for qualitative inspection and future interpretability sections in the manuscript.

## Reproducibility Features

The code includes several reproducibility-oriented mechanisms:

- fixed random seed (`SEED = 42`);
- NumPy, Python, and PyTorch seed control;
- deterministic cuDNN configuration when available;
- structured run directories with serialized configurations;
- filesystem-based recovery of completed jobs.

These choices improve experimental repeatability, although exact reproducibility may still depend on hardware, library versions, and notebook execution order.

## Current State of the Repository

The repository already contains generated model outputs, figures, and summary tables for multiple datasets, indicating that the pipeline has been used in a substantial experimental campaign rather than only as a prototype.

One detail worth documenting carefully is that [Pipeline.ipynb](/e:/Repos/U-net_cikk/Pipeline.ipynb) and [Pipeline.py](/e:/Repos/U-net_cikk/Pipeline.py) are closely related but not guaranteed to be perfectly synchronized at every moment. For example, the notebook currently appears to use a reduced regime list in at least one place, while the script still exposes the broader set `["e50", "e120", "plateau"]`. For the manuscript, it is advisable to define one canonical experimental configuration and keep the text aligned with that exact version.

## Suggested Paper Framing

For the scientific paper, this pipeline can naturally be described as a spectral modeling study with the following narrative arc:

1. formulate ECG or physiological time-series classification as a supervised multi-class problem;
2. hypothesize that class-relevant information may be present in the spectral structure of the signals;
3. compare conventional convolutional, attention-based, and residual models against Fourier-enhanced variants;
4. evaluate whether FNO blocks improve classification when applied to raw latent signal representations and spectrogram representations;
5. report predictive performance together with parameter-efficiency trade-offs to determine whether any FNO advantage is genuine rather than merely capacity-driven.

## Important Caveats to State Explicitly

- The current experiments re-split the pooled original train and test data into a new 60/20/20 stratified partition.
- Best-model selection is validation-driven, not test-driven, which is methodologically appropriate and should be stated clearly.
- Macro-F1 and balanced accuracy are especially important here because the datasets may be class-imbalanced.
- The main scientific claim should be framed around paired Fourier-versus-non-Fourier comparisons, not only around which architecture achieved the highest absolute score.
- The notebook is still evolving, so any final manuscript should reference a frozen experiment version.

## Short Summary

In summary, the pipeline is an end-to-end research workflow for testing whether Fourier Neural Operator components improve physiological time-series classification. It starts from UCR/UEA-style ECG datasets, applies per-sample normalization and optional time-frequency transformation, trains paired Fourier and non-Fourier model families, performs resumable hyperparameter search, evaluates all runs with multiple metrics, and exports figures and tables intended for scientific reporting. The guiding question throughout the study is whether spectral aspects of the data are meaningful enough that explicitly modeling them improves classification.
