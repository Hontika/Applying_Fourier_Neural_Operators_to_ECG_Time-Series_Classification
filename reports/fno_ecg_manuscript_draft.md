# Applying Fourier Neural Operators to ECG Time-Series Classification

## Highlights

- Fourier Neural Operator (FNO) components are investigated for supervised ECG time-series classification.
- Paired baseline and FNO-enhanced architectures are compared to isolate the effect of spectral modeling.
- The study evaluates whether meaningful class-discriminative structure exists in the spectral representation of ECG signals.
- Raw 1D signal models and spectrogram-based models are both included to compare time-domain and time-frequency approaches.
- Parameter-aware evaluation is used to distinguish genuine spectral-modeling benefit from gains caused only by increased model capacity.

## Abstract

Physiological time-series including ECG signals are most commonly classified with convolutional, recurrent, residual, or attention-based neural networks operating either on raw signals or transformed time-frequency representations. Such architectures may not explicitly exploit global spectral structure that could be relevant for discriminating between classes. This study investigates whether Fourier Neural Operator (FNO) components improve the classification of ECG signals, with the primary focus on the ECG5000 dataset.

The experiment compares several baseline models with corresponding FNO-enhanced variants. These include 1D U-Net, 1D Attention U-Net, ResNet, and spectrogram-based convolutional classifiers alongside models that incorporate FNO layers. This paired approach allows the contribution of spectral modeling to be studied under similar conditions. The models are trained using stratified data splits, per-sample normalization, cross-entropy optimization, and validation-based model selection. Performance metrics used to assess each model are accuracy, macro-F1 score, balanced accuracy, and parameter count.

The central question of the study is not only which architecture achieves the highest score, but whether spectral operators provide a meaningful advantage over comparable non-FNO baselines. The resulting framework provides a reproducible basis for analyzing the role of frequency-domain structure in ECG classification. It also helps to determin whether FNO-based processing is beneficial beyond its original use.

## Keywords

ECG classification; physiological time series; Fourier Neural Operator; spectral modeling; U-Net; ResNet; deep learning; time-series classification

## 1. Introduction

The aim of this study was to investigate whether Fourier Neural Operator (FNO) components can improve the classification of physiological time series, with a primary focus on ECG datasets such as ECG5000. The central question was whether these signals contain spectral structure that is meaningful for class discrimination, and whether explicitly modeling this structure can improve performance beyond standard convolutional, residual, or attention-based architectures.

Although FNOs were originally developed for operator-learning tasks, particularly in the context of partial differential equations, their ability to model global spectral representations makes them a promising candidate for time-series classification. ECG signals contain both local morphological patterns and broader frequency-related characteristics, suggesting that spectral-domain modeling may provide complementary information to conventional time-domain feature extraction.

To test this hypothesis, several baseline model families were implemented together with corresponding FNO-enhanced variants. These paired comparisons were designed to isolate the contribution of Fourier-based spectral processing under comparable preprocessing, training, and evaluation conditions. The study therefore evaluates not only which architecture achieves the highest classification performance, but whether FNO-based components provide a measurable and parameter-efficient advantage over non-Fourier baselines.

The motivation follows a similar direction to recent work applying FNO-based models to real-world time-series classification. In particular, Szekeres et al. introduced CF-ResNet-1D for insect wingbeat sound classification and argued that FNOs, although originally popularized for partial differential equation problems, may also be useful for raw time-series classification because of their ability to process global spectral representations. Their work reported that a Fourier-enhanced residual architecture could outperform spectrogram-based models in insect wingbeat classification. The present study transfers this general question from insect wingbeat audio to physiological ECG-like signals: if spectral structure is informative in biological time series, can FNO-based layers help classify ECG signals more effectively?

## 2. Related Work and Motivation

Fourier Neural Operators were originally developed as neural operator models capable of learning mappings between function spaces, particularly in scientific machine learning and partial differential equation contexts. Their practical appeal comes from applying learned transformations to low-frequency Fourier modes, enabling global interactions that are not restricted to local convolutional receptive fields.

For time-series classification, this idea is attractive because many signals contain both local temporal morphology and broader spectral patterns. ECG signals, for example, are often interpreted through waveform shape, rhythm, intervals, and frequency-related characteristics. A purely local convolutional model may learn many useful representations, but it is not explicitly designed to manipulate frequency-domain components. This motivates the use of FNO-style layers as a mechanism for testing whether global spectral structure contributes to classification.

The study is therefore positioned between two common approaches:

- raw-signal deep learning, where 1D models learn directly from the time-domain sequence;
- time-frequency classification, where signals are transformed into spectrogram-like representations before classification.

By including both raw-signal FNO models and spectrogram-based models, the study examines whether spectral information is helpful as an input representation, as an internal learned operator, or both.

## 3. Materials and Data

The pipeline reads datasets stored in UCR/UEA `.ts` format under `Data/<DATASET_NAME>/`. The available datasets are:

- `ECG5000`
- `NonInvasiveFetalECGThorax1`
- `NonInvasiveFetalECGThorax2`

Each record is treated as a univariate time series with an associated class label. Labels are mapped to integer class indices using the training label set, and all test labels are checked against this mapping.

The current experimental setup concatenates the original train and test files and creates a new stratified split:

- 60% training
- 20% validation
- 20% test

This repartitioning should be clearly stated in the final paper because it means the results are not directly equivalent to evaluations performed on the original UCR/UEA train-test split.

## 4. Preprocessing

Each signal is normalized independently using per-sample z-score normalization:

\[
\hat{x} = \frac{x - \mu(x)}{\sigma(x) + \varepsilon}
\]

where \(\mu(x)\) and \(\sigma(x)\) are computed for each individual sample.

The raw 1D dataset class also supports random temporal shifting with zero padding. In the current configuration, this augmentation is disabled by default. For spectrogram-based experiments, signals are transformed with a short-time Fourier transform (STFT), converted to magnitude, and log-compressed before being passed to 2D convolutional classifiers.

## 5. Model Architectures

The model set was chosen to support paired Fourier and non-Fourier comparisons.

### 5.1 Compact 1D U-Net and U-Net-FNO

The compact 1D U-Net is used as a convolutional encoder-decoder baseline. It uses 1D convolutional blocks, downsampling, upsampling, skip connections, and an adaptive average pooling classification head.

The U-Net-FNO variant inserts FNO blocks into the bottleneck. The FNO block applies a real-valued fast Fourier transform along the temporal dimension, keeps a limited number of spectral modes, applies learned complex-valued transformations to those modes, and reconstructs the signal representation through an inverse transform. This directly tests whether spectral manipulation of the latent representation improves classification.

### 5.2 Attention U-Net and Attention U-Net-FNO

The Attention U-Net adds attention gates to the skip connections. This provides a stronger non-Fourier baseline by allowing the decoder to selectively use encoder features.

The Attention U-Net-FNO variant combines attention-gated skip connections with an FNO bottleneck. This comparison is important because it asks whether the Fourier component remains useful even when the baseline already includes an attention mechanism.

### 5.3 ResNet-Based Models

The pipeline includes residual 1D classifiers as additional baselines. These models provide a comparison point outside the U-Net family and help determine whether observed FNO effects are specific to encoder-decoder models or more general across architectures.

The residual model family includes:

- `ResNet1DClassifier`
- `ResNet9_1DClassifier`
- `CFResNet9_1DClassifier`
- `FNOResNet9_1DClassifier`

The Fourier-enhanced residual variants are especially relevant because they follow the same general motivation as CF-ResNet-1D-style architectures: combine local convolutional processing with spectral-domain processing.

### 5.4 Spectrogram CNN and Spectrogram CNN-FNO

The spectrogram CNN processes STFT-derived time-frequency representations with 2D convolutional layers. This model tests whether manually exposing frequency information to the network improves classification.

The spectrogram CNN-FNO extends this idea by adding 2D spectral blocks in the spectrogram domain. This tests whether learned Fourier operators still add value after the input has already been transformed into a time-frequency representation.

## 6. Experimental Design

The training loop uses cross-entropy loss and the AdamW optimizer. Models are selected based on validation performance, and the selected checkpoint is then evaluated on the held-out test set.

The grid search explores shared hyperparameters such as base channel width, learning rate, weight decay, and dropout. For FNO-enhanced models, the search additionally varies the number of retained Fourier modes and the number of FNO layers. For spectrogram models, STFT parameters such as `n_fft` and `hop_length` are also varied.

The pipeline supports fixed-epoch training regimes and plateau-based early stopping. Runs are saved in a structured directory hierarchy:

`models/<DATASET_NAME>/<model_family>/<training_regime>/run_xxx/`

Each run stores the model weights, configuration, and metrics. This makes the search resumable and allows all completed experiments to be reloaded for later evaluation.

## 7. Evaluation Metrics

The main evaluation metrics are:

- test accuracy;
- macro-F1 score;
- balanced accuracy;
- parameter count.

Macro-F1 and balanced accuracy are especially important because ECG and physiological datasets may contain class imbalance. Parameter count is included because the study is not only interested in absolute performance, but also in whether FNO-enhanced models improve performance efficiently.

The most important comparison is not simply the best model overall. The core analysis should compare each Fourier-enhanced model against the closest non-Fourier counterpart:

- U-Net vs U-Net-FNO;
- Attention U-Net vs Attention U-Net-FNO;
- ResNet-style baselines vs Fourier-enhanced residual variants;
- Spectrogram CNN vs Spectrogram CNN-FNO.

## 8. Expected Results Presentation

The final results section should be organized around the study question. A recommended structure is:

### 8.1 Overall model ranking

Report the best-performing model from each family using macro-F1, balanced accuracy, test accuracy, and parameter count.

Suggested table source:

- `models/<DATASET_NAME>/eval_best_overall_per_family.csv`

### 8.2 Paired Fourier vs non-Fourier comparison

Compare baseline and FNO-enhanced models directly. This section should answer whether the FNO versions consistently improve performance, and whether the improvement is large enough to be meaningful.

Suggested comparisons:

- `base_unet` against `unet_fno`
- `att_unet` against `att_unet_fno`
- `resnet9` against `cf_resnet9` and `fno_resnet9`
- `spec_cnn` against `spec_cnn_fno`

### 8.3 Parameter-matched comparison

Use parameter-matched tables to determine whether FNO improvements remain after controlling approximately for model size.

Suggested table source:

- `models/<DATASET_NAME>/param_matched_cross_family.csv`
- `tables/param_matched_cross_family.tex`

### 8.4 Spectrogram versus raw-signal analysis

Compare raw 1D models to spectrogram-based models. This section should clarify whether spectral information is more useful as a precomputed representation, as an internal learned operator, or as a combination of both.

### 8.5 Error analysis and class-level behavior

Use confusion matrices and per-class F1 plots to determine whether FNO-enhanced models improve all classes uniformly or primarily help difficult minority classes.

Suggested figure sources:

- `figs/<DATASET_NAME>/confusion_<family>.jpg`
- `figs/<DATASET_NAME>/per_class_f1*.jpg`

## 9. Discussion

The discussion should interpret the results in terms of the spectral hypothesis. If FNO-enhanced models outperform their baselines, this would suggest that spectral structure is not only present in the data but can be exploited by learned spectral operators. If improvements are inconsistent or limited to certain architectures, this would suggest that the usefulness of FNO components depends on model context, data scale, or parameter budget.

Several outcomes are possible:

- If raw-signal FNO models outperform raw-signal baselines, FNO layers likely capture useful global temporal-frequency information.
- If spectrogram models outperform raw-signal models, explicit time-frequency preprocessing may be more important than learned spectral operators.
- If spectrogram CNN-FNO improves over spectrogram CNN, spectral operators may remain useful even after STFT transformation.
- If FNO models only improve when they have more parameters, the benefit may be architectural capacity rather than spectral modeling itself.
- If FNO models perform similarly to baselines with fewer or comparable parameters, they may provide a more efficient representation for this classification problem.

The key contribution of the study is therefore not merely another ECG classifier, but an empirical test of whether Fourier-based neural operators provide useful inductive bias for physiological time-series classification.

## 10. Limitations

The current pipeline is still in development, and the final paper should reference a frozen version of the notebook or exported script. The current repartitioning strategy uses a new stratified 60/20/20 split over the pooled data rather than the original benchmark split, which should be made explicit.

The study also depends on hyperparameter search breadth. If some model families are searched more extensively than others, this may bias the comparison. Final claims should therefore rely on the aggregate evaluation tables and parameter-aware comparisons rather than isolated best runs.

## 11. Conclusion

This study investigates whether Fourier Neural Operator components improve ECG time-series classification and whether spectral aspects of physiological signals are meaningful for discriminating between classes. The pipeline implements a paired comparison framework in which standard convolutional, attention-based, residual, and spectrogram-based models are evaluated alongside Fourier-enhanced counterparts.

By combining raw-signal models, spectrogram models, FNO bottlenecks, Fourier-enhanced residual blocks, and parameter-aware evaluation, the study provides a structured way to determine whether spectral modeling offers a genuine advantage. The final interpretation should focus on whether FNO-enhanced models consistently outperform comparable baselines, whether those gains persist under parameter matching, and whether the results support the broader hypothesis that ECG classification benefits from explicit spectral-domain processing.

## Data and Code Availability

The experiments use publicly available UCR/UEA-style datasets stored locally under the repository `Data/` directory. The training pipeline, model definitions, evaluation utilities, generated figures, and result tables are stored in the project repository.

## Reference Anchor

This draft is stylistically and structurally inspired by the paper:

Szekeres, B. J., Gyongyossy, M. N., and Botzheim, J. (2025). Applying Fourier Neural Operator to insect wingbeat sound classification: Introducing CF-ResNet-1D. Ecological Informatics, 86, 103055. https://doi.org/10.1016/j.ecoinf.2025.103055

The present study should be positioned as a related investigation in physiological time-series classification rather than insect wingbeat classification, using the shared motivation that Fourier Neural Operators may provide useful global spectral representations for real-world time-series tasks.
