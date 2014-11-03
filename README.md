ceres-mle
=========

A maximum likelihood estimator using Google's [ceres-solver](http://ceres-solver.org/) minimizer.

Works with version `1.10.0rc1` of ceres.

Recently, ceres was updated to allow for unconstrained optimization of arbitrary functions.
This means that ceres can now be used for implementing Maximum Likelihood estimators that make use of the excellent L-BFGS implementation in ceres.

This repository currently only contains a simple demo that fits a normal distribution to a random dataset.

A python script is provided that displays the data and fit.

