# TS Benchmarking Subset

Time series analysis is a central ML problem. In this rapidly evolving domain, new algorithms frequently claim state-of-the-art performance based on particular datasets. However focusing just on one dataset may lead to over-fitting for it’s general characteristics. In this project we apply a special benchmarking methodology to facilitate a fair and robust comparison of time series classification algorithms, thereby advancing evaluation practices.
Namely, we propose a way of selecting a small representative subset of datasets, which allows us to approximate the quality of each model on all available datasets.

So, the goal of the project is to find an efficient subset of datasets that gives an approximation to some benchmark calculated on a large number of datasets. The minimum subset of datasets should give approximately the same ranks (in terms of, for example, correlation) for the model predictions.

Our contribution include:
  - Selection of a core subset of datasets for efficient quality evaluation of time series classification algorithms  and hyperparameters tuning. 
  - Selection of the highest-performing algorithms from a collection of commonly used methods, utilizing a systematic aggregation of metrics.
  - Exploration of the connection between particular dataset characteristics and the quality of classification, as well as the identification of clusters of datasets that share similar characteristics. 

The repository includes the 112 datasets for benchmarking from the University of California, Riverside (UCR) archive. Datasets are stored in the `data/` directory.

