# pairwise-regression

This is a pairwise regression model for predicting a target value of interested based on nonlinear interactions between input features and a key variable. The key variable is a special input feature that is particularly critical for the target value of interest. 

For instance, when predicting a child's weight as a target value of interest, the key predictor may be the child's age. 

The module allows you to input a design matrix of multiple input features, designate a key variable among the features, and then fit a regression model learning the relationship and nonlinear interactions between feature and key variable.

