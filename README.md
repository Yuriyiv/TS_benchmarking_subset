# TS Benchmarking Subset

Time series analysis is a central machine learning problem. In this rapidly evolving domain, new algorithms frequently claim state-of-the-art performance based on particular datasets. However, focusing on a single dataset may lead to overfitting to its specific characteristics. In this project, we apply a special benchmarking methodology to facilitate a fair and robust comparison of time series classification algorithms, thereby advancing evaluation practices.

We propose a way of selecting a small representative subset of datasets, which allows us to approximate the quality of each model on all available datasets efficiently.

## Table of Contents

- [Introduction](#introduction)
  - [Contributions](#contributions)
  - [Installation](#installation)
- [Methodology](#methodology)
  - [Core Dataset Selection via Clustering](#core-dataset-selection-via-clustering)
  - [Feature Extraction](#feature-extraction)
  - [Algorithm Ranking Analysis](#algorithm-ranking-analysis)
- [Datasets, Metrics, and Models](#datasets-metrics-and-models)
  - [Types of Time Series Datasets](#types-of-time-series-datasets)
  - [Time Series Classification Models](#time-series-classification-models)
- [Experiments](#experiments)
- [Repository Structure](#repository-structure)
- [References](#references)

## Introduction

The proliferation of time series data has led to the development of numerous classification algorithms. Evaluating these algorithms fairly requires benchmarking on diverse datasets. However, running every model on all available datasets is computationally intensive and impractical.

Our goal is to find an efficient subset of datasets that provides an approximation to benchmarks calculated on a large number of datasets. The minimum subset should yield approximately the same ranks (e.g., in terms of correlation) for model predictions as the full set.

### Contributions

- **Selection of a Core Subset of Datasets**: We identify a representative subset of datasets for efficient evaluation of time series classification algorithms and hyperparameter tuning.
- **Systematic Aggregation of Metrics**: We select high-performing algorithms from commonly used methods using a systematic aggregation of metrics.
- **Exploration of Dataset Characteristics**: We explore the connection between specific dataset characteristics and classification quality, identifying clusters of datasets with similar features.

The repository includes 112 datasets for benchmarking from the University of California, Riverside (UCR) archive. Datasets are stored in the `papers/datasets/` directory.

### Installation

To set up the project environment and install dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Yuriyiv/TS_benchmarking_subset.git
   cd TS_benchmarking_subset
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Methodology

### Core Dataset Selection via Clustering

We employ clustering techniques to select a core subset of datasets:

1. **Feature Mapping**: We create two feature maps for our datasets:
   - **Simple Features**: Basic characteristics such as length, seasonality, dimensionality, class imbalance, and the number of classes.
   - **Tsfresh Features**: Advanced features extracted using the `tsfresh` library, resulting in 783 features per dataset.

2. **Dimensionality Reduction**: We use UMAP to embed the high-dimensional feature space into 2D for visualization and clustering.

3. **Clustering**: Clusters are formed using HDBSCAN, an algorithm capable of detecting clusters of varying densities.

4. **Representative Selection**: In each cluster, we select a representative dataset closest to the cluster centroid. The set of all representatives forms our core dataset.

5. **Evaluation Metrics**: We assess the effectiveness of the core dataset using:
   - **Mean Absolute Error (MAE)**
   - **Mean Squared Error (MSE)**
   - **Kendall Rank Correlation**
   - **Spearman Rank Correlation**

**Results of Clustering with Simple Features:**

- **Number of Clusters**: 21
- **Selected Core Datasets**:
  - BeetleFly
  - Crop
  - DistalPhalanxTW
  - ElectricDevices
  - EthanolLevel
  - Fish
  - FordA
  - HouseTwenty
  - InsectWingbeatSound
  - ItalyPowerDemand
  - MiddlePhalanxOutlineCorrect
  - MixedShapesRegularTrain
  - OliveOil
  - Plane
  - PowerCons
  - ShapesAll
  - SmallKitchenAppliances
  - SwedishLeaf
  - ToeSegmentation1
  - UWaveGestureLibraryAll
  - Yoga

**Evaluation Metrics:**

| Features     | MAE    | MSE     | Kendall | Spearman |
|--------------|--------|---------|---------|----------|
| Simple       | 1.1762 | 1.9129  | 0.8756  | 0.9764   |
| Tsfresh      | 1.8778 | 4.7208  | 0.8319  | 0.9579   |

### Feature Extraction

We utilized the `tsfresh` library to automatically extract features from the datasets, resulting in a total of 783 features. The most significant features identified are:

- **Standard Deviation**
- **Variance**
- **FFT Coefficients** (Real and Imaginary parts for specific coefficients)
- **FFT Angle Coefficients**

We also identified simple attributes that are easy to compute and effectively describe the datasets:

- **Entropy**
- **Gini Impurity**
- **Number of Classes**
- **Dataset Size**

### Algorithm Ranking Analysis

We analyzed how dataset features influence the ranking of classification algorithms:

- **Approach**: We split datasets based on specific features (e.g., dataset size, number of classes) and compared the rankings of algorithms on these subsets.
- **Findings**:
  - **Dataset Size**: Models like `1NN-DTW` and `CNN` performed differently on small vs. large datasets.
  - **Number of Classes**: Algorithms showed varying performance when datasets had two classes compared to more than two.

**Effect of Dataset Size on Model Rankings:**

| Model       | Rank (Size ≤ 737) | Rank (Size > 737) | Significant Difference |
|-------------|-------------------|-------------------|------------------------|
| 1NN-DTW     | 1.50              | 2.83              | Yes                    |
| CNN         | 3.17              | 1.57              | Yes                    |
| BOSS        | 15.93             | 8.63              | Yes                    |
| ...         | ...               | ...               | ...                    |

**Effect of Number of Classes on Model Rankings:**

| Model       | Rank (2 Classes) | Rank (>2 Classes) | Significant Difference |
|-------------|------------------|-------------------|------------------------|
| 1NN-DTW     | 1.03             | 2.30              | Yes                    |
| BOSS        | 15.03            | 9.17              | Yes                    |
| EE          | 7.57             | 9.90              | Yes                    |
| ...         | ...              | ...               | ...                    |

## Datasets, Metrics, and Models

### Types of Time Series Datasets

We categorized the datasets into various domains to ensure diversity:

| Domain                         | Count | Datasets                                                                                                                                            |
|---------------------------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Audio                          | 2     | InsectWingbeat, Phoneme                                                                                                                              |
| Devices                        | 11    | Computers, ElectricDevices, LargeKitchenAppliances, RefrigerationDevices, ScreenType, SmallKitchenAppliances, ACSF1, FreezerRegularTrain, FreezerSmallTrain, PowerCons, HouseTwenty |
| Electro-cardiography            | 7     | CinCECGTorso, ECG200, ECG5000, ECGFiveDays, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, TwoLeadECG                                       |
| Electrical Penetration Graphs   | 2     | InsectEPGRegularTrain, InsectEPGSmallTrain                                                                                                           |
| Electro-oculography             | 2     | EOGHorizontalSignal, EOGVerticalSignal                                                                                                               |
| Human Activity Recognition      | 11    | CricketX, CricketY, CricketZ, GunPoint, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, UWaveGestureLibraryAll, UWaveGestureLibraryX, UWaveGestureLibraryY, UWaveGestureLibraryZ |
| Hemodynamics                    | 3     | PigAirWayPressure, PigArtPressure, PigCVP                                                                                                            |
| Images                         | 31    | Adiac, ArrowHead, BeetleFly, BirdChicken, DiatomSizeReduction, DistalPhalanxOutlineAgeGroup, DistantPhalanxOutlineCorrect, DistalPhalanxTW, FaceAll, FaceFour, FacesUCR, FiftyWords, HandOutlines, Herring, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, OSULeaf, PhalangesOutlineCorrect, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ShapesAll, SwedishLeaf, Symbols, WordSynonyms, Yoga, MixedShapesRegularTrain, MixedShapesSmallTrain, Crop |
| Motions                        | 6     | Haptics, InlineSkate, ToeSegmentation1, ToeSegmentation2, Worms, WormsTwoClass                                                                       |
| Sensors                        | 13    | Car, Earthquakes, FordA, FordB, ItalyPowerDemand, Lightning2, Lightning7, MoteStrain, Plane, SonyAIBORRobotSurface1, SonyAIBORRobotSurface2, Trace, Wafer |
| Spectrograms                   | 10    | Beef, Coffee, Ham, Meat, OliveOil, Strawberry, Wine, EthanolLevel, SemgHandGenderCh2, SemgHandMovementCh2, SemgHandSubjectCh2, Rock                   |
| Simulated                      | 8     | CBF, ChlorineConcentration, Mallat, ShapeletSim, SyntheticControl, TwoPatterns, BME, SmoothSubspace, UMD                                              |
| Traffic                   | 1     | Chinatown                                                                                                                                                  |

### Time Series Classification Models

We used various time series classification models across different categories:

| Model               | Model Type                | Source                                                                                                                      |
|---------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| 1NN-DTW             | Nearest Neighbor          | [Link](https://www.cs.ucr.edu/~eamonn/KAIS_2004_warping.pdf)                                                              |
| Arsenal             | Ensemble                  | [Link](https://arxiv.org/abs/2104.07551)                                                                                   |
| BOSS                | Dictionary-Based          | [Link](https://www.researchgate.net/publication/282271002_The_BOSS_is_concerned_with_time_series_classification_in_the_presence_of_noise) |
| Catch22             | Feature-Based             | [Link](https://arxiv.org/abs/1901.10200)                                                                                   |
| cBOSS               | Dictionary-Based          | [Link](https://arxiv.org/abs/1907.11815)                                                                                   |
| CIF                 | Tree-Based                | [Link](https://arxiv.org/abs/2008.09172)                                                                                   |
| CNN                 | Deep Learning             | [Link](https://arxiv.org/abs/1611.06455)                                                                                   |
| DrCIF               | Tree-Based                | [Link](https://arxiv.org/abs/2305.01429)                                                                                   |
| EE                  | Feature-Based             | [Link](https://link.springer.com/article/10.1007/s10618-014-0361-2)                                                        |
| FreshPRINCE         | Deep Learning             | [Link](https://arxiv.org/abs/2201.12048)                                                                                   |
| HC1                 | Hierarchical Classification | [Link](https://ieeexplore.ieee.org/document/7837946)                                                                         |
| HC2                 | Hierarchical Classification | [Link](https://arxiv.org/abs/2104.07551)                                                                                   |
| Hydra               | Ensemble                  | [Link](https://arxiv.org/abs/2301.02152)                                                                                   |
| Hydra-MR            | Ensemble                  | [Link](https://arxiv.org/abs/2301.02152)                                                                                   |
| InceptionT          | Deep Learning             | [Link](https://arxiv.org/abs/1909.04939)                                                                                   |
| Mini-R              | Unclassified              | [Link](https://arxiv.org/abs/2012.08791)                                                                                   |
| MrSQM               | Shapelet-Based            | [Link](https://www.sktime.net/en/v0.25.0/api_reference/auto_generated/sktime.classification.shapelet_based.MrSQM.html#sktime.classification.shapelet_based.MrSQM) |
| Multi-R             | Multi-Resolution          | [Link](https://arxiv.org/abs/2102.00457)                                                                                   |
| PF (Proximity Forest)| Tree-Based               | [Link](https://arxiv.org/abs/1808.10594)                                                                                   |
| RDST                | Dictionary-Based          | [Link](https://arxiv.org/pdf/2109.13514)                                                                                   |
| RISE                | Interval-Based Ensemble   | [Link](https://www.sktime.net/en/v0.25.0/api_reference/auto_generated/sktime.classification.interval_based.RandomIntervalSpectralEnsemble.html#sktime.classification.interval_based.RandomIntervalSpectralEnsemble) |
| ROCKET              | Transformation-Based      | [Link](https://arxiv.org/abs/1910.13051)                                                                                   |
| RSF                 | Tree-Based                | [Link](https://link.springer.com/article/10.1007/s10618-016-0473-y)                                                        |
| RSTSF               | Interval-Based Forest     | [Link](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.interval_based.RSTSF.html)       |
| ResNet              | Deep Learning             | [Link](https://arxiv.org/abs/1611.06455)                                                                                   |
| ShapeDTW            | Similarity-Based          | [Link](https://arxiv.org/abs/1606.01601)                                                                                   |
| Signatures          | Feature-Based             | [Link](https://www.sktime.net/en/v0.25.0/api_reference/auto_generated/sktime.classification.feature_based.SignatureClassifier.html#sktime.classification.feature_based.SignatureClassifier) |
| STC                 | Shapelet-Based            | [Link](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.shapelet_based.ShapeletTransformClassifier.html) |
| STSF                | Interval-Based Forest     | [Link](https://www.sktime.net/en/v0.25.0/api_reference/auto_generated/sktime.classification.interval_based.SupervisedTimeSeriesForest.html#sktime.classification.interval_based.SupervisedTimeSeriesForest) |
| TDE                 | Dictionary-Based          | [Link](https://arxiv.org/abs/2105.03841)                                                                                   |
| TS-CHIEF            | Ensemble                  | [Link](https://arxiv.org/abs/1906.10329)                                                                                   |
| TSF                 | Tree-Based                | [Link](https://www.sktime.net/en/v0.25.0/api_reference/auto_generated/sktime.classification.interval_based.TimeSeriesForestClassifier.html) |
| TSFresh             | Feature-Based             | [Link](https://www.sciencedirect.com/science/article/pii/S0925231218304843)                                                  |
| WEASEL              | Feature-Based             | [Link](https://arxiv.org/abs/1701.07681)                                                                                   |
| WEASEL-D            | Feature-Based             | [Link](https://arxiv.org/pdf/2301.10194)                                                                                   |

## Experiments

The general scheme of our experiments is as follows:

1. **Data Collection**: We downloaded the metrics, datasets, and models from the [UCR Time Series Classification Archive](https://timeseriesclassification.com/results/PublishedResults/).

2. **Baseline Ranking**: Models were ranked based on their performance across all datasets to establish a baseline.

3. **Feature Extraction**: We extracted features using both simple methods and the `tsfresh` library.

4. **Clustering and Core Dataset Selection**: We performed clustering to select representative datasets.

5. **Model Evaluation on Core Subset**: Models were re-evaluated using only the core subset to compare against the baseline rankings.

6. **Analysis**: We analyzed the trade-offs between efficiency and accuracy when using the reduced dataset selection.

## Repository Structure

```
├── notebooks
│   ├── Arrangement_for_TS_benchmarking.ipynb
│   ├── DatasetParsing.ipynb
│   ├── FeatureStatistics.ipynb
│   ├── Task4.ipynb
│   ├── features
│   │   ├── features.csv
│   │   └── tsfresh_important_features.csv
│   └── images
│       ├── Kendall_all.png
│       ├── MAE_all.png
│       ├── MSE_all.png
│       ├── Spearman_all.png
│       ├── kde_plot_entropy.png
│       ├── kde_plot_gini.png
│       ├── kde_plot_number_of_classes.png
│       ├── kde_plot_size.png
│       ├── umaped_clustered_tsfresh.png
│       └── umaped_simple_tsfresh.png
├── papers
│   ├── datasets [exists only locally after downloading via script]
│   │   ├── [Dataset JSON files]
│   └── metrics
│       ├── Bakeoff2017
│       ├── Bakeoff2021
│       ├── Bakeoff2023
│       └── HIVE-COTEV2
└── src
    ├── __init__.py
    ├── data_loader.py
    └── data_stats.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

- **notebooks/**: Contains Jupyter notebooks for data preparation, feature extraction, and analysis.
  - **features/**: CSV files with extracted features.
  - **images/**: Visualizations like KDE plots and clustering results.
- **papers/**: Contains datasets and benchmarking metrics.
  - **datasets/**: JSON files of datasets.
  - **metrics/**: Benchmarking results from various papers.
- **src/**: Source code modules for data loading and statistical computations.

## References

1. **Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)**  
   Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). *Neurocomputing*, 307, 72–77.  
   [Link](https://doi.org/10.1016/j.neucom.2018.03.067)

2. **The UEA Multivariate Time Series Classification Archive, 2018**  
   Bagnall, A., Dau, H. A., Lines, J., Flynn, M., Large, J., Bostrom, A., ... & Keogh, E. (2018).  
   [Link](https://arxiv.org/abs/1811.00075)

3. **Bake off redux: A review and experimental evaluation of recent time series classification algorithms, 2024**
   Middlehurst, M., Schäfer, P., & Bagnall, A. (2024).  *Data Mining and Knowledge Discovery*, 38(4), 1958–2031.
   [Link](https://doi.org/10.1007/s10618-024-01022-1)


