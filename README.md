# Credit Prediction
## Credit Limitation Prediction using Polynomial Regression and Random Forest Regressor

## Introduction
In the dynamic domain of financial technology, the precision in predicting credit limits is pivotal for tailoring banking services to individual needs. Our project, 'Credit Limit Prediction,' leverages machine learning to forecast credit limits with unprecedented accuracy. We've prepared a comprehensive dataset of user features, preprocessed to build a model that truly reflects financial behaviors.

## Models Used
We employ two machine learning models, **Polynomial Regression** and **Random Forest**, chosen for their ability to discern non-linear patterns and complex decision boundaries in credit limit prediction. An extensive evaluation was performed, focusing on hyperparameter tuning and its effects on the loss function and the $$ R^2 $$ score, leading us to the sweet spot of model complexity.

## Computational Performance
A key aspect of our study was comparing the performance of model training on **GPUs** versus **CPUs**. This not only showcased computational benefits but also provided insights into the deployment of machine learning models in practical settings.

## Objective
The goal of this document is to outline our methodology, share our findings, and discuss the implications of our work within the scope of credit limit prediction. We aspire to enhance the predictive analytics tools available to the financial industry, thereby facilitating more nuanced credit decisions.

## Methodology
Our project's methodology is built on advanced machine learning techniques and computational frameworks to predict credit limits accurately. The approach includes:

1. **Polynomial Regression Implementation**:
   - Utilized the JAX framework for JIT compilation to efficiently train on a GPU.
   - Transformed the dataset into polynomial features using Scikit-Learn.
   - Adopted functional programming paradigms to manage complexity and enhance maintainability.
   - Training JAX polynomial regression model both on CPU and GPU , to compare the results of training on different hardwares
   - Employed Stochastic Gradient Descent with momentum for better optimization and faster convergence. We also compared using different momentum values on the final results
   - We compare the differnet results in using random forest and polynomial regression for multivariable regression , as well as comparing the result of using different polynomial degree for dataset on the final results

2. **Random Forest Model**:
   - Applied the polynomial feature dataset to the Random Forest model, conducted exclusively on a CPU due to scikit-learn's limitations.
   - We set different values for hyperparameters to check their effect on final results 


The dataset consistency across both models ensured a fair comparison, with 9,000 training examples and 1,160 test examples, each with 17 features. The application of polynomial features expanded the feature space, enriching the dataset for our predictive models. The Random Forest model outperformed Polynomial Regression, indicating its superior ability to detect complex patterns in the dataset.


## Scripts 
Our implementation includes three main files :

- **data_preprocessing.py** : Includes the loading data file and preprocessing it to create a simple clean dataset for feeding the models . consist of categorizing some of the features and filling NaN values
- **regression_model.py** : Includes the scripts of implementation of polynomial regression with JAX
- **random_forest_model.py** : Includes the scripts of implementation of random forest regression with SciKit-Learn

