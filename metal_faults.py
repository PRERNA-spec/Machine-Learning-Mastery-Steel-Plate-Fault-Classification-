path = r"C:\Users\User\Desktop\metals\faults.csv"

import pandas as pd
data = pd.read_csv(path)

print(data.isnull().sum())
from sklearn.preprocessing import StandardScaler

# List numerical columns
num_cols = data.columns.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults'])

# Standardize
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Encode 'TypeOfSteel_A300' and 'TypeOfSteel_A400' as binary features
data = pd.get_dummies(data, columns=['TypeOfSteel_A300', 'TypeOfSteel_A400'], drop_first=True)

from sklearn.model_selection import train_test_split

# Define features and target
X = data.drop(['Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults'], axis=1)
y = data['Z_Scratch']


from sklearn.model_selection import train_test_split
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(32, 18))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# Plot distributions of selected features
for col in ['X_Minimum', 'Y_Minimum', 'Pixels_Areas']:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Train a logistic regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# Evaluate
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



from sklearn.ensemble import RandomForestClassifier

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf.predict(X_test)

# Evaluate
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



from sklearn.svm import SVC

# Train an SVM classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Predict on test set
y_pred_svm = svm.predict(X_test)

# Evaluate
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))



from sklearn.model_selection import cross_val_score

# Cross-validation scores for Random Forest
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
print('Random Forest CV Scores:', cv_scores_rf)
print('Mean CV Score:', cv_scores_rf.mean())



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Random Forest')
plt.show()



# Feature importance from Random Forest
importances = rf.feature_importances_
feature_names = X.columns
# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance for Random Forest')
plt.show()



