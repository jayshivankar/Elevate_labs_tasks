# AI & ML Internship Tasks

This repository contains my work for the AI & ML internship tasks focused on **Data Cleaning & Preprocessing** (Task 1), **Exploratory Data Analysis** (Task 2), **Linear Regression** (Task 3), **Classification with Logistic Regression** (Task 4), **Decision Trees & Random Forests** (Task 5), **K-Nearest Neighbors** (Task 6), **Support Vector Machines** (Task 7), and **K-Means Clustering** (Task 8).

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

## 🌳 Task 5: Decision Trees and Random Forests

### Objective
Learn and implement tree-based models for classification tasks using ensemble methods.

### What I Did
- Used Heart Disease dataset to predict cardiovascular disease presence.
- Trained Decision Tree Classifier and visualized the tree structure using Graphviz.
- Analyzed overfitting by experimenting with tree depth and pruning parameters.
- Implemented Random Forest classifier and compared performance with single decision tree.
- Evaluated models using cross-validation and various classification metrics.
- Interpreted feature importances to identify key health indicators.
- Compared model complexity vs performance trade-offs between single tree and ensemble.

### Tools Used
- Python, Scikit-learn, Pandas, Matplotlib, Graphviz

---

## 📍 Task 6: K-Nearest Neighbors (KNN) Classification

### Objective
Implement and understand instance-based learning using KNN algorithm for multi-class classification.

### What I Did
- Used the Iris dataset to classify flower species based on sepal and petal measurements.
- Performed feature normalization using StandardScaler for optimal KNN performance.
- Implemented KNeighborsClassifier with different K values (1, 3, 5, 7, 10).
- Evaluated model performance using accuracy score and confusion matrix.
- Visualized decision boundaries for different K values to understand model behavior.
- Analyzed the elbow method to determine optimal K value for balancing bias-variance tradeoff.
- Compared Euclidean vs Manhattan distance metrics and their impact on model performance.

### Tools Used
- Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 🎯 Task 7: Support Vector Machines (SVM)

### Objective
Implement SVM classifiers with different kernels and understand margin maximization concepts.

### What I Did
- Used Breast Cancer dataset for binary classification using SVM.
- Implemented both linear SVM and non-linear SVM with RBF kernel.
- Visualized decision boundaries and support vectors in 2D feature space.
- Tuned hyperparameters (C, gamma) using GridSearchCV for optimal performance.
- Compared model performance between linear and RBF kernels.
- Analyzed the effect of regularization parameter C on margin width and misclassification.
- Evaluated models using cross-validation and classification metrics.
- Explored the kernel trick for handling non-linearly separable data.

### Tools Used
- Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 🎪 Task 8: Clustering with K-Means

### Objective
Perform unsupervised learning using K-Means clustering for customer segmentation.

### What I Did
- Used Mall Customer Segmentation dataset to identify customer groups.
- Performed feature scaling and applied PCA for 2D visualization of clusters.
- Implemented K-Means clustering with multiple K values to find optimal clusters.
- Used Elbow Method and Silhouette Score to determine optimal number of clusters (K=5).
- Visualized clusters with color-coding and analyzed cluster characteristics.
- Interpreted cluster centers to understand different customer segments.
- Evaluated clustering quality using inertia and silhouette analysis.
- Compared different initialization methods (K-means++ vs random).

### Tools Used
- Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---


