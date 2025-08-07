#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#supress warnings
import warnings
warnings.filterwarnings("ignore")

# Sklearn libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

#statmodel libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn import tree

#miscellaneous
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('dataset.csv')


# In[3]:


df.shape


# In[4]:


categorical_cols = []

for col in df.columns:
    unique_vals = df[col].nunique()
    dtype = df[col].dtype

    if dtype == 'object' or dtype.name == 'category' or unique_vals < 15:
        categorical_cols.append(col)

print("Categorical Columns:", categorical_cols)


# In[5]:


numeric_cols = []

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_cols.append(col)

print("Numeric Columns:", numeric_cols)


# In[6]:


df.drop(columns=['Index'], inplace=True)


# In[7]:


import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Custom transformer to add WHO flags
class WHOFlags(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['is_ph_safe'] = ((X['pH'] >= 6.5) & (X['pH'] <= 8.5)).astype(int)
        X['is_nitrate_safe'] = (X['Nitrate'] <= 50).astype(int)
        X['is_iron_safe'] = (X['Iron'] <= 0.3).astype(int)
        X['is_chloride_safe'] = (X['Chloride'] <= 250).astype(int)
        X['is_manganese_safe'] = (X['Manganese'] <= 0.4).astype(int)
        X['is_sulfate_safe'] = (X['Sulfate'] <= 500).astype(int)
        X['is_tds_safe'] = (X['Total Dissolved Solids'] < 1000).astype(int)
        X['is_turbidity_safe'] = (X['Turbidity'] < 5).astype(int)
        X['is_conductivity_safe'] = (X['Conductivity'] < 1400).astype(int)
        X['is_chlorine_safe'] = (X['Chlorine'] <= 5).astype(int)
        X['is_copper_safe'] = (X['Copper'] <= 2).astype(int)
        return X

# Define columns
numeric_features = [
    'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Turbidity', 'Fluoride',
    'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese',
    'Total Dissolved Solids', 'Water Temperature', 'Air Temperature', 'Day', 'Time of Day'
]
categorical_features = ['Color', 'Source', 'Month']

# Sample data for prototyping
df_sampled = df.sample(n=100000, random_state=42)

X = df_sampled.drop(columns=['Target'])
y = df_sampled['Target']

# Pipelines for numeric and categorical features
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

full_pipeline = Pipeline([
    ('whoflags', WHOFlags()),
    ('preprocessor', preprocessor)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Fit pipeline and transform training data
full_pipeline.fit(X_train)

X_train_processed = full_pipeline.transform(X_train)

# Get feature names properly
numeric_feature_names = numeric_features  # unchanged by numeric pipeline
cat_onehot = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
categorical_feature_names = cat_onehot.get_feature_names_out(categorical_features)

feature_names = list(numeric_feature_names) + list(categorical_feature_names)

# Convert to DataFrame for SHAP or analysis
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)

# Train Random Forest
model = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42,class_weight='balanced')
model.fit(X_train_processed, y_train)

# Transform and create DataFrame for test set
X_test_processed = full_pipeline.transform(X_test)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Test accuracy
accuracy = model.score(X_test_processed, y_test)
print(f"Test accuracy: {accuracy:.4f}")


# In[8]:


print(y_train.value_counts())


# In[9]:


# Assuming:
# - model is your trained RandomForestClassifier
# - feature_names is the list of feature names after preprocessing (numeric + one-hot encoded categorical)

import numpy as np

# Get feature importances
importances = model.feature_importances_

# Sort features by importance descending
indices = np.argsort(importances)[::-1]

# Pick top 3
top3_indices = indices[:3]
top3_features = [(feature_names[i], importances[i]) for i in top3_indices]

print("Top 3 important features (from RF model):")
for feat, imp in top3_features:
    print(f"{feat}: {imp:.4f}")


# In[10]:


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load your pipeline and model (adjust paths)
full_pipeline = joblib.load('full_pipeline.joblib')
model = joblib.load('rf_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df_input = pd.DataFrame([data])

    # Preprocess input
    X_processed = full_pipeline.transform(df_input)

    # Predict probability and class
    proba_all = model.predict_proba(X_processed)[0]
    proba = proba_all[1] if len(proba_all) > 1 else proba_all[0]
    pred = int(proba >= 0.5)

    # Get feature names after preprocessing
    numeric_features = full_pipeline.named_steps['preprocessor'].transformers_[0][2]
    cat_onehot = full_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
    categorical_features = cat_onehot.get_feature_names_out(full_pipeline.named_steps['preprocessor'].transformers_[1][2])
    feature_names = list(numeric_features) + list(categorical_features)

    # Get feature importances from model
    importances = model.feature_importances_

    # Create pandas Series for easier sorting
    feat_importance_series = pd.Series(importances, index=feature_names)

    # Get top 3 important features overall
    top3 = feat_importance_series.sort_values(ascending=False).head(3)

    top_features = [{'feature': feat, 'importance': float(top3[feat])} for feat in top3.index]

    return jsonify({
        'prediction': pred,
        'probability': float(proba),
        'top_features': top_features
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)


# In[ ]:


app.run(port=5000, debug=True, use_reloader=False)


# In[13]:


import joblib
joblib.dump(model, "rf_model.joblib")

