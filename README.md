# Fraud-Detection-Logistic-Regression

README for Fraud Detection Using Logistic Regression

Title:

Fraud Detection Using Logistic Regression

Introduction:

This project focuses on building a fraud detection model using the Logistic Regression algorithm. The objective is to classify fraudulent and non-fraudulent transactions by exploring data patterns, handling class imbalances, and evaluating model performance. The analysis highlights practical challenges and provides insights to enhance fraud detection systems.

Objectives
1.	Perform Exploratory Data Analysis (EDA) to understand data structure and distributions.
2.	Preprocess data by handling class imbalance using SMOTE and addressing outliers.
3.	Develop and evaluate a Logistic Regression model to classify fraudulent transactions.

Tools and Technologies
•	Programming Language: Python
•	Libraries: pandas, scikit-learn, seaborn, matplotlib, SMOTE
•	Dataset: Fraudulent transaction dataset with binary classification target.

Key Findings
1.	Class Imbalance: The dataset was imbalanced with 700 non-fraudulent and 300 fraudulent transactions. SMOTE improved the model’s ability to detect fraud.
2.	Predictive Model: Logistic Regression achieved an accuracy of 59% and precision of 41% after balancing the dataset.
3.	Challenges: Moderate positive correlation among numerical predictors impacted the performance of the Logistic Regression model.

Steps to Reproduce
1.	Perform EDA to inspect data structure and detect missing values.
2.	Preprocess the data by normalizing numerical data, handling outliers, and balancing classes using SMOTE.
3.	Split the data into training and testing sets, and train the Logistic Regression model.
4.	Evaluate the model using accuracy, precision, confusion matrix, and gains chart.

File Descriptions
1.	fraud_detection.py: Python script for data preprocessing, modeling, and evaluation.
2.	fraud_dataset.csv: Dataset used for analysis.
3.	Fraud_Detection_Report.docx: Detailed project report outlining methodology, findings, and conclusions.
4.	README.docx: Documentation for project overview.

Contact Information
Author: Emmanuel Popoola
Email: N01511@humber.ca

