# PM2.5 Air Pollution Prediction using Gaussian Process Regression

## Goal

In this project, we aim to help a city predict and audit the concentration of fine particulate matter (PM2.5) per cubic meter of air. In an initial phase, the city has collected preliminary measurements using mobile measurement stations. The goal is now to develop a pollution model that can predict the air pollution concentration in locations without measurements. This model will then be used to determine particularly polluted areas where permanent measurement stations should be deployed.

A pervasive class of models for weather and meteorology data are **Gaussian Processes (GPs)**. We are implementing **Gaussian Process regression** in order to model air pollution and try to predict the concentration of PM2.5 at previously unmeasured locations.

> For a gentle introduction to GPs: [Intro to Gaussian Process Regression](link_to_intro)

## Problem Set-up and Challenges

### Features:
- **Inputs:** Coordinates (X, Y) of the city map
- **Target:** PM2.5 pollution concentration at a given location

### Challenges:

- **Model Selection:** Determination of the right kernel and hyperparameters is key for GP performance.
- **Large Scale Learning:** As the number of observations increases, the computational cost of GPs grows exponentially. The posterior complexity is O(n^3). To address this, several approaches can be applied, such as undersampling, kernel low-rank approximations, and Local GPs. In this project, we use **Local GPs**, where various individual GPs are fitted to specific regions of the city map instead of a global regressor for the entire dataset.
- **Asymmetric Cost:** Cost-sensitive learning is implemented with a loss function that treats different types of errors differently:

\[
\ell_W(f(x), \hat{f}(x)) = (f(x) - \hat{f}(x))^2 \cdot
\begin{cases} 
25, & \text{if } f(x) \leq \hat{f}(x) \text{ (underprediction)} \\
1, & \text{if } f(x) \leq 1.2 \cdot f(x) \text{ (slight overprediction)} \\
10, & \text{if } 1.2 \cdot f(x) \leq \hat{f}(x) \text{ (significant overprediction)} 
\end{cases}
\]

## Approach and Results

The main contribution is the division of the city map into various squares of the same size, where an individual GP is fitted to each. By using **Local GPs**, we can make predictions that are more specific to every location.

This approach achieves a more specific prediction than implementing a global GP and produces a cumulative cost of **48.519**.

## References
- Gaussian Processes Tutorial: [Intro to Gaussian Process Regression](link_to_intro)
