# Robust Predictive Modeling: Regularization in High-Dimensional Data

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## 📌 Overview

In modern machine learning, deploying models that generalize to unseen data is critical. When dealing with high-dimensional datasets where the feature space is dense relative to the number of samples, Ordinary Least Squares (OLS) regression models inherently memorize training noise, leading to severe overfitting. 

This repository contains a comprehensive demonstration of how to implement, tune, and evaluate **L1 (Lasso)** and **L2 (Ridge)** regularization techniques to build robust predictive models. We approach this through a realistic financial services business scenario: predicting Customer Churn. 

## 🏢 Business Context & Dataset

* **Problem Formulation:** Predicting a continuous `Churn_Score` (0-100) for a financial institution's customer base to identify at-risk accounts.
* **The Dataset:** The working data contains 400 customer samples spanning 100 distinct features (including demographics, account information, transactions, engagement metrics, and behavioral noise). 
* **The Challenge:** With a feature-to-sample ratio of 25%, this is a moderately high-dimensional problem where standard linear regression is guaranteed to overfit.

## 🛠️ Technical Stack

The accompanying Jupyter Notebook is engineered using standard Python data science ecosystem libraries:
* **NumPy (v2.1.3)** & **Pandas (v2.2.3)** for vectorization and DataFrame manipulation.
* **Scikit-Learn** for model architecture, data scaling (`StandardScaler`), and cross-validation.
* **Matplotlib** & **Seaborn** (`seaborn-v0_8-darkgrid` style) for regularization path visualization.

## 🧠 Core Methodologies Explored

### 1. The Overfitting Baseline (OLS)
We establish a baseline using standard Linear Regression without penalty terms. 
* **Observation:** The model heavily overfits, achieving a Training R² of 0.7971 but degrading to a Testing R² of 0.5187 (a gap of nearly 28%).

### 2. Ridge Regression (L2 Regularization)
Ridge introduces an L2 penalty (proportional to the sum of squared coefficients), enforcing coefficient shrinkage without eliminating features.
* **Impact:** Ideal for this dataset's multicollinearity, providing more stable coefficients and slightly improving the test baseline (Testing R²: 0.5228 at $\alpha = 1.0$).
* **Hyperparameter Tuning:** We execute a logarithmic path search to find the mathematically optimal penalty, locating the best $\alpha = 75.4312$.

### 3. Lasso Regression (L1 Regularization)
Lasso introduces an L1 penalty (proportional to the sum of absolute coefficients), inducing sparsity by forcing irrelevant feature coefficients strictly to zero.
* **Impact:** Performs automated feature selection. At $\alpha = 0.1$, the model identified and eliminated 14 noisy features, utilizing only 86 predictors. 
* **Result:** Achieved the most robust generalization with a Testing R² of 0.5977, significantly outperforming both the unpenalized and L2 baselines.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```
---

## 🤝 Contributing

Contributions to improve the examples, add new functions or methods, or fix typos are always welcome. Please feel free to open an issue or submit a pull request!

---

## Connect with me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abhay-kumar-sharma-a22a94171)

