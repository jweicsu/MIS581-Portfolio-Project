# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import statsmodels.api as sm
from scipy import stats


# Load the dataset (assuming it’s in CSV format)
df = pd.read_csv('HR_Employee_Attrition.csv')

# Convert categorical variables to numerical (if not already done)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# For simplicity, focus on a few variables, including Job Satisfaction, Monthly Income, Work-Life Balance
df['JobSatisfaction'] = pd.to_numeric(df['JobSatisfaction'])
df['WorkLifeBalance'] = pd.to_numeric(df['WorkLifeBalance'])
df['MonthlyIncome'] = pd.to_numeric(df['MonthlyIncome'])

# Drop any missing values
df = df.dropna()


# Feature selection and scaling
X = df[['JobSatisfaction']]  # You can include other factors like 'MonthlyIncome' or 'WorkLifeBalance'
y = df['Attrition']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# Predict probabilities for test data
y_prob = log_model.predict_proba(X_test_scaled)[:,1]

# Generate plot for logistic regression
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_prob, color='red', label='Predicted Probability', linewidth=2)
plt.xlabel('Job Satisfaction')
plt.ylabel('Attrition Probability')
plt.title('Logistic Regression of Job Satisfaction and Attrition')
plt.legend()
plt.grid(True)
plt.show()


# Select relevant features for correlation matrix
correlation_data = df[['Attrition', 'JobSatisfaction', 'WorkLifeBalance', 'MonthlyIncome']]

# Calculate the correlation matrix
corr_matrix = correlation_data.corr()

# Plot the correlation matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Key Variables')
plt.show()


# Perform a Chi-Square test between JobSatisfaction and Attrition
contingency_table = pd.crosstab(df['JobSatisfaction'], df['Attrition'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Print Chi-Square results
print(f"Chi-Square value: {chi2}")
print(f"P-value: {p}")

# Visualize the Chi-Square test results with a bar plot
contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Chi-Square Test of Job Satisfaction and Attrition')
plt.xlabel('Job Satisfaction')
plt.ylabel('Employee Count')
plt.legend(title='Attrition', loc='upper right')
plt.show()


# Get predictions and calculate accuracy
y_pred = log_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Logistic Regression Model: {accuracy}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


