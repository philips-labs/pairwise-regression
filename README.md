# pairwise-regression

This is a pairwise regression model for predicting a target value of interested based on nonlinear interactions between input features and a key variable. The key variable is a special input feature that is particularly critical for the target value of interest. 

For instance, when predicting a child's weight as a target value of interest, the key predictor may be the child's age. 

The module allows you to input a design matrix of multiple input features, designate a key variable among the features, and then fit a regression model learning the relationship and nonlinear interactions between feature and key variable.

# Usage
A usage example with sample data is given [here]() for reference.

# Requirements
'''
Python 3.7.3
numpy==1.16.1
scikit-learn==0.21.2
matplotlib==3.1.0
'''

# Contributing
We appreciate any contributions in the form of ideas, feature requests, bug reports, bug fixes, documentation improvements, code reformatting, and code submissions. Please see the [Contributing guide](https://github.com/philips-internal/pairwise-regression/blob/main/CONTRIBUTING.md).