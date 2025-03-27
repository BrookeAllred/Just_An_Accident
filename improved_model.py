import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def create_interaction_features(df):
    # Create interaction between important numerical features
    df['ps_ind_01_03'] = df['ps_ind_01'] * df['ps_ind_03']
    df['ps_ind_01_04'] = df['ps_ind_01'] * df['ps_ind_04_cat']
    df['ps_ind_03_04'] = df['ps_ind_03'] * df['ps_ind_04_cat']
    return df

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['ps_ind_01', 'ps_ind_03', 'ps_calc_01', 'ps_calc_02', 
                     'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                     'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10',
                     'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']

# Apply feature engineering to train and test data
train_data = create_interaction_features(train_data)
test_data = create_interaction_features(test_data)

# Scale numerical features
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# Prepare features and target
X = train_data.drop(['id', 'target'], axis=1)
y = train_data['target']

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with cross-validation
models = {
    'Logistic Regression (L2)': LogisticRegression(penalty='l2', C=0.1, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}

# Evaluate models using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f'{name} - Mean ROC AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})')

# Train final model on full training data
final_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
final_model.fit(X, y)

# Make predictions on test data
test_features = test_data.drop('id', axis=1)
predictions = final_model.predict_proba(test_features)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'target': predictions
})
submission.to_csv('submission_improved.csv', index=False)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close() 