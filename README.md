# AI/ML Internship - Task 1: Dataset Exploration & Visualization

Welcome to my AI/ML internship repository! This project covers the foundational Exploratory Data Analysis (EDA) and visualization tasks.

---

## Task Objective
Perform **Exploratory Data Analysis (EDA)** and **Data Visualization** on the classic Iris dataset to analyze feature distributions, check statistical properties, and uncover patterns between flower species.

##  Dataset Overview
* **Name:** Iris Dataset (Loaded via Seaborn)
* **Size:** 150 rows, 5 columns (**Zero missing values**)
* **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
* **Target:** `species` (*Setosa*, *Virginica*, *Versicolor*)

---

## Workflow & Steps

### 1. Setup & Inspection
* **Libraries:** `pandas`, `numpy`, `seaborn`, `matplotlib`
* **Exploration:** Used `df.head()`, `df.info()`, and `df.describe()` for structural and statistical summaries.

### 2. Data Visualization
Applied three distinct statistical plots using Seaborn & Matplotlib:
* **Scatter Plot:** Analyzed the relationship between sepal length and width across different species.
* **Histogram (with KDE):** Examined the distribution and density frequency of `sepal_length`.
* **Box Plot:** Visualized the spread, median, and outliers of `sepal_length` across the three species.

---

## Key Findings
* **Statistical Insights:** The average sepal length is **5.84 cm** (ranging from 4.3 cm to 7.9 cm).
* **Clear Class Separation:** *Iris setosa* forms a completely distinct cluster, making it easily separable from the other two species based on sepal characteristics.
* **Feature Spread:** *Iris virginica* displays the highest variance and the maximum values for sepal length.

#  AI/ML Internship - Task 2: Stock Price Prediction Using Linear Regression

This project focuses on predicting stock prices using historical market data fetched from Yahoo Finance (`yfinance`) and building a simple predictive model using Scikit-Learn.

---

##  Task Objective
The primary objective is to build a **Linear Regression model** that utilizes daily market features (`Open`, `High`, `Low`, `Volume`) to predict the next day's closing price (`Next_Close`) for a given stock ticker (AAPL).

##  Dataset Overview
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Ticker Used:** `AAPL` (Apple Inc.)
* **Time Period:** 2024-01-01 to 2026-01-01
* **Features ($X$):** `Open`, `High`, `Low`, `Volume`
* **Target ($y$):** `Next_Close` (The shifted `Close` price of the next trading day)

---

##  Workflow & Steps

### 1. Data Fetching & Feature Engineering
* **Data Retrieval:** Downloaded historical data smoothly using `yf.download`.
* **Target Creation:** Used `.shift(-1)` on the `Close` column to create the target variable for next-day prediction and dropped the resulting row with missing values (`.dropna()`).

### 2. Model Training
* **Data Splitting:** Divided the dataset into training and testing sets (80% train, 20% test) using `train_test_split` with a fixed `random_state=42` for reproducibility.
* **Algorithm:** Trained a standard `LinearRegression` model from `sklearn.linear_model`.

### 3. Evaluation & Visualization
* **Predictions:** Generated target predictions using the test set (`model.predict()`).
* **Visual Validation:** Plotted a time-series line graph using Matplotlib to overlay the **Actual Prices** against the **Predicted Prices** to evaluate the model's tracking performance.

---

##  Key Visual Output
* **Actual vs Predicted Line Plot:** The final plot maps the actual values versus model predictions over time, showing how closely the linear trends capture the shifting dynamics of the stock price.

* ---

## AI/ML Internship - Task 3: Heart Disease Prediction Using Logistic Regression

This project focuses on predicting the presence of heart disease in patients based on a set of clinical health features and metrics, utilizing Scikit-Learn to build a highly accurate classification pipeline.

### Task Objective
The primary objective is to build a Logistic Regression classification model to predict whether a patient has heart disease (`target` = 1) or not (`target` = 0), perform full Exploratory Data Analysis (EDA), apply robust preprocessing, and evaluate using advanced classification metrics.

### Dataset Overview
* **Data Source:** `HeartDiseaseTrain-Test.csv`
* **Target Variable:** `target` (0 = No Heart Disease, 1 = Heart Disease Presence)
* **Key Features:** `age`, `sex`, `chest_pain_type`, `resting_blood_pressure`, `cholestoral`, `Max_heart_rate`, `oldpeak`, `vessels_colored_by_flourosopy`, `thalassemia`, etc.

### Workflow & Steps

#### 1. Data Cleaning & Advanced Preprocessing
* **Categorical Encoding:** Applied `pd.get_dummies(drop_first=True)` to dynamically handle text/categorical columns (like `sex`, `chest_pain_type`, `thalassemia`) into standard numeric format to avoid the dummy variable trap.
* **Feature Scaling:** Used Scikit-Learn's `StandardScaler` to normalize and standardize continuous numeric variables (`age`, `cholestoral`, `resting_blood_pressure`, `Max_heart_rate`, `oldpeak`), ensuring the optimizer converges efficiently without scaling bias.

#### 2. Exploratory Data Analysis (EDA)
* **Target Distribution:** Evaluated target class balance using count plots.
* **Feature Distributions:** Plotted custom histograms with Kernel Density Estimation (KDE) for continuous variables like age and cholesterol.
* **Correlation Analysis:** Generated a correlation heatmap for numeric features to study continuous linear trends against the target variable.

#### 3. Model Training & Evaluation
* **Data Splitting:** Executed an 80% Train and 20% Test split using `train_test_split` with stratification to maintain class proportions.
* **Model Training:** Fit a standard `LogisticRegression` classification model with `max_iter=1000`.
* **Metrics Assessment:** Evaluated the model using standard metrics:
  * **Accuracy Score:** Achieved a high performance of **94.31%** accuracy on unseen test data.
  * **ROC Curve & AUC:** Plotted the Receiver Operating Characteristic curve, achieving an outstanding **AUC score of 0.94**, validating high discriminative power.
  * **Confusion Matrix:** Plotted a clean confusion matrix heatmap to inspect True Positives, True Negatives, and misclassifications.

#### 4. Feature Importance Extraction
* Extracted the model coefficients (`model.coef_[0]`) sorted cleanly by their absolute magnitude to highlight the health factors impacting predictions the most (e.g., specific `chest_pain_type` classes and fluoroscopy vessel counts).
