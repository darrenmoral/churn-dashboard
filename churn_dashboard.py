import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load model and metadata
model = joblib.load('churn_model.pkl')
features = joblib.load('model_features.pkl')
dataset = pd.read_csv('CustomerData.csv')

# Ensure required columns and correct order
def prepare_features(df, features):
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features].copy()
    # Convert all to numeric if needed
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

X = prepare_features(dataset, features)

# Predictions
dataset['Churn Probability'] = model.predict_proba(X)[:, 1]
dataset['Predicted Churn'] = model.predict(X)

# Title
st.title("Customer Churn Dashboard")

# Section: Show data
@st.cache_data
def load_data(path: str):
    data = pd.read_csv('CustomerData.csv')
    return data

st.subheader("Customer Data")
df = load_data('CustomerData.csv')
st.write(df)

# Section: Tenure Distribution Histogram (Descriptive Visualization)
st.subheader("Tenure Distribution: Churned vs Not Churned")

fig3, ax3 = plt.subplots(figsize=(10, 6))
df_churned = dataset[dataset['Churn'] == 'Yes']
df_not_churned = dataset[dataset['Churn'] == 'No']

ax3.hist(df_not_churned['tenure'], bins=20, alpha=0.7, label='Not Churned', edgecolor='black')
ax3.hist(df_churned['tenure'], bins=20, alpha=0.7, label='Churned', edgecolor='black')

ax3.set_title("Tenure Distribution by Churn")
ax3.set_xlabel("Tenure (months)")
ax3.set_ylabel("Number of Customers")
ax3.legend()
st.pyplot(fig3)

y_true = dataset['Churn'].map({'Yes': 1, 'No': 0})
y_pred = dataset['Predicted Churn'].map({'Yes': 1, 'No': 0})

# Section 2: Churn Risk
st.subheader("Top Customers at Risk of Churn")
top_risk = dataset[['customerID', 'Churn Probability']].sort_values(by='Churn Probability', ascending=False).head(10)
st.dataframe(top_risk)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top_risk, x='Churn Probability', y='customerID', palette='Reds_r', ax=ax)
ax.set_title("Top 10 At-Risk Customers")
st.pyplot(fig)

# Section 3: Feature Importance
st.subheader("Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax2)
st.pyplot(fig2)

# Section 4: Upload for Prediction
st.subheader("Predict New Customers")
uploaded = st.file_uploader("Upload a CSV file with customer data", type="csv")
if uploaded:
    new_data = pd.read_csv(uploaded)

    # Handle missing one-hot columns
    for col in features:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[features]

    # Predict
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]
    new_data['Churn Prediction'] = predictions
    new_data['Churn Probability'] = probabilities

    st.subheader("📋 Prediction Results")
    st.dataframe(new_data[['Churn Prediction', 'Churn Probability'] + features])
