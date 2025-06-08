# micro_IT1
import pandas as pd

df = pd.read_csv('xAPI-Edu-Data.csv')
df.head()
from google.colab import sheets
sheet = sheets.InteractiveSheet(df=df)
print(f"Number of duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(df)}")
import numpy as np

numerical_cols = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column '{col}': Found {len(outliers)} outliers.")
    from sklearn.preprocessing import StandardScaler
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = [col for col in categorical_cols if col != 'Class']
target = df['Class']
df_categorical_encoded = pd.get_dummies(df[categorical_features], drop_first=True)
scaler = StandardScaler()
df_numerical_scaled = scaler.fit_transform(df[numerical_cols])
df_numerical_scaled = pd.DataFrame(df_numerical_scaled, columns=numerical_cols, index=df.index)
df_processed = pd.concat([df_categorical_encoded, df_numerical_scaled, target], axis=1)
display(df_processed.head())
from sklearn.model_selection import train_test_split
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
log_reg_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)
log_reg_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

print("Logistic Regression model trained.")
print("Random Forest model trained.")
print("SVM model trained.")
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2']
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

log_reg = LogisticRegression(random_state=42, solver='liblinear')
rf_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

print("Modules imported and parameter grids defined.")

grid_search_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='f1_macro')
grid_search_lr.fit(X_train, y_train)
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr = best_lr_model.predict(X_val)
accuracy_lr = accuracy_score(y_val, y_pred_lr)
precision_lr = precision_score(y_val, y_pred_lr, average='macro')
recall_lr = recall_score(y_val, y_pred_lr, average='macro')
f1_lr = f1_score(y_val, y_pred_lr, average='macro')

print("Logistic Regression Tuning and Evaluation Complete.")
print(f"Best LR Params: {grid_search_lr.best_params_}")
print(f"LR Validation Metrics: Accuracy={accuracy_lr:.4f}, Precision={precision_lr:.4f}, Recall={recall_lr:.4f}, F1-score={f1_lr:.4f}")

# Perform GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(rf_clf, param_grid_rf, cv=5, scoring='f1_macro')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Evaluate the best Random Forest model on the validation set
y_pred_rf = best_rf_model.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_pred_rf)
precision_rf = precision_score(y_val, y_pred_rf, average='macro')
recall_rf = recall_score(y_val, y_pred_rf, average='macro')
f1_rf = f1_score(y_val, y_pred_rf, average='macro')

print("\nRandom Forest Tuning and Evaluation Complete.")
print(f"Best RF Params: {grid_search_rf.best_params_}")
print(f"RF Validation Metrics: Accuracy={accuracy_rf:.4f}, Precision={precision_rf:.4f}, Recall={recall_rf:.4f}, F1-score={f1_rf:.4f}")

# Perform GridSearchCV for SVM
grid_search_svm = GridSearchCV(svm_clf, param_grid_svm, cv=5, scoring='f1_macro')
grid_search_svm.fit(X_train, y_train)
best_svm_model = grid_search_svm.best_estimator_

# Evaluate the best SVM model on the validation set
y_pred_svm = best_svm_model.predict(X_val)
accuracy_svm = accuracy_score(y_val, y_pred_svm)
precision_svm = precision_score(y_val, y_pred_svm, average='macro')
recall_svm = recall_score(y_val, y_pred_svm, average='macro')
f1_svm = f1_score(y_val, y_pred_svm, average='macro')

print("\nSVM Tuning and Evaluation Complete.")
print(f"Best SVM Params: {grid_search_svm.best_params_}")
print(f"SVM Validation Metrics: Accuracy={accuracy_svm:.4f}, Precision={precision_svm:.4f}, Recall={recall_svm:.4f}, F1-score={f1_svm:.4f}")

# Store the best models and their performance
tuned_models = {
    'Logistic Regression': {'model': best_lr_model, 'metrics': {'accuracy': accuracy_lr, 'precision': precision_lr, 'recall': recall_lr, 'f1_score': f1_lr}},
    'Random Forest': {'model': best_rf_model, 'metrics': {'accuracy': accuracy_rf, 'precision': precision_rf, 'recall': recall_rf, 'f1_score': f1_rf}},
    'SVM': {'model': best_svm_model, 'metrics': {'accuracy': accuracy_svm, 'precision': precision_svm, 'recall': recall_svm, 'f1_score': f1_svm}}
}
# Compare performance on the validation set
print("\nTuned Model Performance on Validation Set:")
for model_name, result in tuned_models.items():
    metrics = result['metrics']
    print(f"{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")

# Determine the best performing model based on F1-score (or another chosen metric)
best_model_name = None
best_f1 = -1

for model_name, result in tuned_models.items():
    if result['metrics']['f1_score'] > best_f1:
        best_f1 = result['metrics']['f1_score']
        best_model_name = model_name

print(f"\nBest performing model on the validation set based on F1-score: {best_model_name}")

# The best model is stored in tuned_models[best_model_name]['model']
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the best Logistic Regression model on the testing set
best_lr_model = tuned_models['Logistic Regression']['model']
y_pred_lr_test = best_lr_model.predict(X_test)
accuracy_lr_test = accuracy_score(y_test, y_pred_lr_test)
precision_lr_test = precision_score(y_test, y_pred_lr_test, average='macro')
recall_lr_test = recall_score(y_test, y_pred_lr_test, average='macro')
f1_lr_test = f1_score(y_test, y_pred_lr_test, average='macro')

print("Logistic Regression Test Metrics:")
print(f"  Accuracy: {accuracy_lr_test:.4f}")
print(f"  Precision: {precision_lr_test:.4f}")
print(f"  Recall: {recall_lr_test:.4f}")
print(f"  F1-score: {f1_lr_test:.4f}")

# Evaluate the best Random Forest model on the testing set
best_rf_model = tuned_models['Random Forest']['model']
y_pred_rf_test = best_rf_model.predict(X_test)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
precision_rf_test = precision_score(y_test, y_pred_rf_test, average='macro')
recall_rf_test = recall_score(y_test, y_pred_rf_test, average='macro')
f1_rf_test = f1_score(y_test, y_pred_rf_test, average='macro')

print("\nRandom Forest Test Metrics:")
print(f"  Accuracy: {accuracy_rf_test:.4f}")
print(f"  Precision: {precision_rf_test:.4f}")
print(f"  Recall: {recall_rf_test:.4f}")
print(f"  F1-score: {f1_rf_test:.4f}")

# Evaluate the best SVM model on the testing set
best_svm_model = tuned_models['SVM']['model']
y_pred_svm_test = best_svm_model.predict(X_test)
accuracy_svm_test = accuracy_score(y_test, y_pred_svm_test)
precision_svm_test = precision_score(y_test, y_pred_svm_test, average='macro')
recall_svm_test = recall_score(y_test, y_pred_svm_test, average='macro')
f1_svm_test = f1_score(y_test, y_pred_svm_test, average='macro')

print("\nSVM Test Metrics:")
print(f"  Accuracy: {accuracy_svm_test:.4f}")
print(f"  Precision: {precision_svm_test:.4f}")
print(f"  Recall: {recall_svm_test:.4f}")
print(f"  F1-score: {f1_svm_test:.4f}")

# Compare performance on the test set
test_results = {
    'Logistic Regression': {'accuracy': accuracy_lr_test, 'precision': precision_lr_test, 'recall': recall_lr_test, 'f1_score': f1_lr_test},
    'Random Forest': {'accuracy': accuracy_rf_test, 'precision': precision_rf_test, 'recall': recall_rf_test, 'f1_score': f1_rf_test},
    'SVM': {'accuracy': accuracy_svm_test, 'precision': precision_svm_test, 'recall': recall_svm_test, 'f1_score': f1_svm_test}
}

print("\nTuned Model Performance on Test Set:")
for model_name, metrics in test_results.items():
    print(f"{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")

# Determine the best performing model on the test set based on F1-score
best_test_model_name = None
best_test_f1 = -1

for model_name, metrics in test_results.items():
    if metrics['f1_score'] > best_test_f1:
        best_test_f1 = metrics['f1_score']
        best_test_model_name = model_name

print(f"\nBest performing model on the test set based on F1-score: {best_test_model_name}")
