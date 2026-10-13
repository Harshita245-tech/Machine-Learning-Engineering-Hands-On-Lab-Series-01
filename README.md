# 🤖 Machine Learning Engineering – Hands-On Lab Series 01
 
**Technologies:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

---

## 📌 Project Overview

This repository contains the implementation of five machine learning exercises covering supervised and unsupervised learning. The exercises demonstrate the complete workflow of data inspection, preprocessing, model training, evaluation, diagnostic visualization, feature-importance analysis, and business interpretation.

The project uses Python and Scikit-learn to build and evaluate machine learning models for insurance cost prediction, bank marketing conversion prediction, interpretable classification, ensemble learning, and wholesale customer segmentation.

### 🎯 Objectives

- Understand and implement five core machine learning algorithms.
- Perform data cleaning, preprocessing, encoding, scaling, and transformation.
- Train and evaluate models using suitable metrics.
- Generate visualizations to understand model performance and data patterns.
- Interpret results and translate findings into practical business recommendations.
- Organize notebook implementations, result visualizations, and report documentation in a GitHub repository.

---

## 📚 Exercise 1: Multiple Linear Regression – Insurance Cost Prediction

### 🎯 Objective
Predict individual medical insurance charges using demographic and lifestyle information.

### 📊 Dataset
The insurance dataset contains features such as:
- `age` – Age of the individual
- `sex` – Sex category
- `bmi` – Body Mass Index
- `children` – Number of dependents
- `smoker` – Smoking status
- `region` – Residential region
- `charges` – Medical insurance expenses

### ⚙️ Implementation
- Inspected the dataset structure, missing values, duplicates, and target distribution.
- Applied numerical preprocessing using imputation and standardization.
- Encoded categorical variables using one-hot encoding.
- Split the dataset into training and testing subsets.
- Trained a Linear Regression model using raw insurance charges.
- Compared raw-target regression with a `log1p`-transformed target model.
- Evaluated predictions using MSE, RMSE, MAE, and R².
- Examined regression coefficients and residual diagnostic plots.

### 📈 Recorded Results

| Metric | Raw Target Model |
|---|---:|
| R² Score | 0.7836 |
| RMSE | $5,796.28 |
| MAE | $4,181.19 |
| MSE | 33,596,915.85 |

The recorded smoker coefficient was approximately $23,651.13, indicating a substantial positive association between smoking status and predicted medical expenses, holding the other included features constant.

### 💡 Key Learning
This exercise demonstrates regression pipelines, categorical encoding, target transformation, coefficient interpretation, and residual-based model diagnostics.

---

## 📚 Exercise 2: Logistic Regression – Bank Marketing Conversion Prediction

### 🎯 Objective
Predict whether a bank customer will subscribe to a term deposit following a direct marketing campaign.

### 📊 Dataset
The bank marketing dataset includes demographic, financial, and campaign-related information, including `age`, `job`, `balance`, `duration`, `campaign`, `pdays`, `previous`, and `poutcome`.

The target variable is `y`, representing whether the customer subscribed.

### ⚙️ Implementation
- Inspected class distribution and identified the imbalance between subscription outcomes.
- Encoded categorical features and prepared numerical features.
- Created a stratified training/testing split.
- Trained Logistic Regression with class weighting.
- Generated predicted probabilities for the subscribed class.
- Compared the default probability threshold of 0.50 with a threshold of 0.35.
- Evaluated the classifier using ROC-AUC, precision, recall, F1-score, and a confusion matrix.
- Examined the relationship between threshold selection and campaign outreach.

### 📈 Recorded Results

| Metric | Recorded Result |
|---|---:|
| ROC-AUC | 0.9085 |
| Recall at threshold 0.35 | 0.9017 |
| Precision at threshold 0.35 | 0.3343 |
| F1-score at threshold 0.35 | 0.4877 |
| Accuracy at threshold 0.35 | 0.7785 |

At a threshold of 0.35, the confusion matrix recorded 7,607 true negatives, 2,374 false positives, 130 false negatives, and 1,192 true positives.

### 💡 Key Learning
This exercise demonstrates binary classification, class imbalance handling, probability-based predictions, ROC-AUC evaluation, and threshold tuning for business-oriented decision-making.

---

## 📚 Exercise 3: Decision Trees – Interpretable Classification

### 🎯 Objective
Build an interpretable classification model and investigate how tree complexity affects predictive performance.

### 📊 Dataset
The notebook implementation uses the Breast Cancer Wisconsin Diagnostic dataset, containing 569 observations, 30 features, and two target classes: malignant and benign.

### ⚙️ Implementation
- Inspected the dataset and target class distribution.
- Split the data into training and testing subsets.
- Compared Decision Trees using Gini impurity and entropy.
- Examined unrestricted trees to understand overfitting.
- Applied cost-complexity pruning using `ccp_alpha`.
- Compared training accuracy, testing accuracy, precision, recall, and F1-score.
- Examined tree depth and extracted readable decision rules.

### 📈 Recorded Results

| Metric | Selected Pruned Tree |
|---|---:|
| Test Accuracy | 0.9386 |
| Precision | 0.9577 |
| Recall | 0.9444 |
| F1-score | 0.9510 |
| Tree Depth | 3 |
| Number of Leaves | 5 |

The selected pruned tree used a `ccp_alpha` value of approximately `0.0076411`.

### 💡 Key Learning
This exercise demonstrates interpretable classification, impurity-based splitting, overfitting analysis, cost-complexity pruning, and extraction of decision rules.

---

## 📚 Exercise 4: Random Forest – Ensemble Learning and Feature Importance

### 🎯 Objective
Build a Random Forest classifier for bank marketing prediction and study its generalization performance and important predictive features.

### 📊 Dataset
The exercise uses the Bank Marketing classification dataset, with the objective of predicting term deposit subscriptions.

### ⚙️ Implementation
- Trained a Random Forest classifier using bootstrap aggregation and randomized feature selection.
- Evaluated out-of-bag (OOB) performance.
- Tracked OOB score convergence as the number of trees increased.
- Performed hyperparameter tuning using GridSearchCV.
- Compared the effects of estimator count, maximum depth, and feature sampling.
- Generated a confusion matrix for the tuned model.
- Compared Mean Decrease in Impurity (MDI) with permutation importance.
- Benchmarked Logistic Regression, Decision Tree, and Random Forest using test F1-score.

### 📈 Recorded Results

| Metric | Tuned Random Forest |
|---|---:|
| OOB Score | 0.9005 |
| Test Accuracy | 0.9014 |
| Precision | 0.7374 |
| Recall | 0.2443 |
| F1-score | 0.3670 |
| ROC-AUC | 0.9243 |

The best recorded grid-search configuration used 200 estimators, `max_features='sqrt'`, and `max_depth=12`.

### 🔎 Feature Importance Findings

**MDI feature importance** ranked `duration` first, followed by `poutcome` and `month`.

**Permutation importance** ranked `poutcome` first, followed by `duration` and `pdays`.

The two methods provide complementary perspectives on how features contribute to model decisions and predictive performance.

### 💡 Key Learning
This exercise demonstrates ensemble modelling, OOB validation, hyperparameter tuning, feature-importance interpretation, and comparative classifier evaluation.

---

## 📚 Exercise 5: K-Means Clustering – Wholesale Customer Segmentation

### 🎯 Objective
Segment wholesale customers according to their annual purchasing patterns across six spending categories.

### 📊 Dataset
The Wholesale Customers dataset contains annual spending information for:
- `Fresh`
- `Milk`
- `Grocery`
- `Frozen`
- `Detergents_Paper`
- `Delicassen`

### ⚙️ Implementation
- Inspected the spending distributions and feature skewness.
- Applied `log1p` transformation to reduce the effect of highly skewed spending values.
- Standardized the transformed features.
- Evaluated candidate cluster counts from K = 2 through K = 8.
- Used the Elbow Method to examine within-cluster sum of squares.
- Used silhouette analysis to compare cluster separation.
- Fitted a four-cluster K-Means model.
- Calculated average annual spending by cluster in the original scale.
- Visualized cluster profiles and translated spending patterns into customer personas.

### 📈 Recorded Results

| Cluster Count | Silhouette Score |
|---|---:|
| K = 2 | 0.2903 |
| K = 3 | 0.2594 |
| K = 4 | 0.1885 |
| K = 5 | 0.1916 |
| K = 6 | 0.2009 |
| K = 7 | 0.1958 |
| K = 8 | 0.1852 |

The final profiling used four clusters to compare customer spending patterns across all six categories.

### 💡 Business Applications
- Develop customer-specific product recommendations.
- Identify customer groups with different purchasing patterns.
- Support inventory planning across product categories.
- Design targeted marketing and customer-retention strategies.
- Inform customer-specific service and credit policies using additional business information.

### 💡 Key Learning
This exercise demonstrates unsupervised learning, logarithmic transformation, feature scaling, cluster-count evaluation, centroid profiling, and business interpretation of customer segments.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming and model implementation |
| Pandas | Dataset manipulation and analysis |
| NumPy | Numerical operations and transformations |
| Matplotlib | Plotting and visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Jupyter Notebook | Interactive development and experimentation |
| LaTeX | Technical report preparation |
| Git and GitHub | Version control and project hosting |

---


---



## 📊 Project Outcomes

Through these five exercises, the project demonstrates:

- End-to-end regression and classification workflows.
- Data preprocessing and feature transformation.
- Evaluation using regression and classification metrics.
- Decision-tree interpretability and pruning.
- Random Forest validation and feature-importance analysis.
- Unsupervised customer segmentation and cluster profiling.
- Visualization-driven interpretation and business recommendations.
- Technical documentation and reproducible project organization.

---

## 🏁 Conclusion

The Machine Learning Engineering Hands-On Lab Series provided practical experience with five fundamental machine learning techniques. The exercises covered medical cost prediction, marketing conversion classification, interpretable decision trees, ensemble-based prediction, and wholesale customer segmentation.

The project strengthened practical understanding of preprocessing pipelines, model evaluation, feature importance, diagnostic visualizations, clustering methods, and business interpretation. The notebook and accompanying report document the implementation process and recorded results for each exercise.

---



