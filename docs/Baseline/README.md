# Baseline Predictors in Collaborative Filtering

## Overview

In **Collaborative Filtering (CF)** systems, baseline predictors (also known as biases) are essential for modeling systematic tendencies in user ratings that are independent of specific user-item interactions. These predictors account for:

- **User Biases *b<sub>u</sub>***: Some users consistently rate items higher or lower than the average.
- **Item Biases *b<sub>i</sub>***: Some items consistently receive higher or lower ratings than the average.

Accurately modeling these biases allows the CF system to isolate and better understand the true interactions between users and items, leading to more precise recommendations.

## Baseline Prediction Formula

The baseline predictor for an unknown rating ***r<sub>ui</sub>*** is denoted by ***b<sub>ui</sub>*** and is calculated as:

***b<sub>ui</sub> = μ + b<sub>u</sub> + b<sub>i</sub>***


Where:
- ***μ*** is the **overall average rating** across all items.
- ***b<sub>u</sub>*** is the **bias of user *u***, representing the deviation of user ***u***'s ratings from the average.
- ***b<sub>i</sub>*** is the **bias of item *i***, representing the deviation of item ***i***'s ratings from the average.

### Example

Consider predicting the rating of the movie *Titanic* by user Joe:
- **Overall average rating *μ***: 3.7 stars.
- **Item bias for *Titanic* *b<sub>i</sub>***: +0.5 stars (Titanic is rated higher than average).
- **User bias for Joe *b<sub>u</sub>***: -0.3 stars (Joe tends to rate lower than average).

The baseline prediction ***b<sub>ui</sub>*** is:

***b<sub>ui</sub>*** = 3.7 - 0.3 + 0.5 = 3.9★  

## Optimization of Bias Parameters

To accurately estimate the user and item biases ***b<sub>u</sub>*** and ***b<sub>i</sub>***, we solve the following **least squares optimization problem** with regularization to prevent overfitting:

\[
\min_{b_u, b_i} \sum_{(u,i) \in K} (r_{ui} - \mu - b_u - b_i)^2 + \lambda_1 \left( \sum_u b_u^2 + \sum_i b_i^2 \right)
\]

Where:
- **K** is the set of all user-item pairs with observed ratings.
- ***λ<sub>1</sub>*** is the **regularization parameter** that controls the magnitude of the biases.  

### Optimization Method

The parameters ***b<sub>u</sub>*** and ***b<sub>i</sub>*** can be efficiently estimated using **Stochastic Gradient Descent (SGD)**.

### Example with Netflix Data

For the Netflix dataset:
- **Mean rating *μ***: 3.6 stars.
- **User biases *b<sub>u</sub>***:
  - **Average**: 0.044
  - **Standard Deviation**: 0.41
  - **Average of absolute values |*b<sub>u</sub>*|**: 0.32
- **Item biases *b<sub>i</sub>***:
  - **Average**: -0.26
  - **Standard Deviation**: 0.48
  - **Average of absolute values |*b<sub>i</sub>*|**: 0.43
## Simplified Calculation Method

An alternative, albeit slightly less accurate, method involves **decoupling** the calculation of ***b<sub>i</sub>*** and ***b<sub>u</sub>***:

1. **Calculate Item Biases *b<sub>i</sub>***:

\[
b_i = \frac{\sum_{u \in R(i)} (r_{ui} - \mu)}{\lambda_2 + |R(i)|}
\]

2. **Calculate User Biases *b<sub>u</sub>***:

\[
b_u = \frac{\sum_{i \in R(u)} (r_{ui} - \mu - b_i)}{\lambda_3 + |R(u)|}
\]

Where:
- ***R(i)*** is the set of all users who have rated item ***i***.
- ***R(u)*** is the set of all items rated by user ***u***.
- ***λ<sub>2</sub>*** and ***λ<sub>3</sub>*** are **regularization parameters** determined via cross-validation.

### Typical Regularization Parameters (Netflix Dataset)

- ***λ<sub>2</sub>*** = 25
- ***λ<sub>3</sub>*** = 10

These parameters help shrink the averages towards zero, mitigating the risk of overfitting.

## Further Improvements

Baseline predictors can be further enhanced by incorporating **temporal dynamics** within the data. This involves considering how user preferences and item popularity evolve over time, leading to more accurate and dynamic bias estimates. Details on this approach are discussed in subsequent sections of the referenced material.

## Conclusion

Baseline predictors are a fundamental component in CF systems, effectively capturing and adjusting for inherent user and item biases. By accurately modeling these biases, CF models can focus on the genuine interactions between users and items, resulting in more reliable and personalized recommendations.

