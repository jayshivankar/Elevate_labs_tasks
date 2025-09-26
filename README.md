# AI & ML Internship Tasks

This repository contains my work for the AI & ML internship tasks focused on **Data Cleaning & Preprocessing** (Task 1), **Exploratory Data Analysis** (Task 2), **Linear Regression** (Task 3), and **Classification with Logistic Regression** (Task 4).

---

## 📂 Task 1: Data Cleaning & Preprocessing

### Objective
Prepare raw data for machine learning by handling missing values, encoding categorical variables, and scaling numerical features.

### What I Did
- Loaded the Titanic dataset and inspected its structure, missing values, and data types.
- Handled missing values in `Age` and `Cabin` using median imputation and mode-based filling.
- Encoded categorical variables like `Sex` and `Embarked` using label encoding and one-hot encoding.
- Scaled numerical features such as `Age` and `Fare` using standardization.
- Detected and visualized outliers using boxplots and applied IQR-based removal.

### Tools Used
- Python, Pandas, NumPy, Matplotlib, Seaborn

---

## 📊 Task 2: Exploratory Data Analysis (EDA)

### Objective
Understand the dataset through statistical summaries and visualizations to identify patterns, trends, and relationships.

### What I Did
- Generated summary statistics (mean, median, standard deviation) for numeric features.
- Created histograms and boxplots to understand distributions and detect skewness.
- Used pairplots and a correlation heatmap to explore feature relationships.
- Identified insights such as the correlation between `Fare` and `Survived`, and the distribution of `Age` across passenger classes.
- Noted potential multicollinearity between `Pclass` and `Fare`.

### Tools Used
- Pandas, Matplotlib, Seaborn, Plotly

---

## 📈 Task 3: Linear Regression

### Objective
Implement and understand simple and multiple linear regression models for predictive analysis.

### What I Did
- Used a housing price dataset to predict prices based on features like square footage, bedrooms, etc.
- Preprocessed data by handling missing values and encoding categorical variables.
- Split data into training and testing sets using train_test_split.
- Implemented both simple linear regression (single feature) and multiple linear regression.
- Evaluated model performance using MAE, MSE, and R² metrics.
- Visualized regression lines and analyzed coefficient interpretations.
- Compared model performance between simple and multiple regression approaches.

### Tools Used
- Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 🔍 Task 4: Classification with Logistic Regression

### Objective
Build and evaluate a binary classifier using logistic regression for medical diagnosis prediction.

### What I Did
- Utilized the Breast Cancer Wisconsin dataset for binary classification (malignant vs benign).
- Performed feature standardization and train-test split (80-20 ratio).
- Implemented logistic regression model using Scikit-learn.
- Evaluated model with confusion matrix, precision, recall, F1-score, and ROC-AUC metrics.
- Visualized the sigmoid function and ROC curve to explain model predictions.
- Experimented with different classification thresholds to optimize performance.
- Handled class imbalance and interpreted feature importance.

### Tools Used
- Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---


