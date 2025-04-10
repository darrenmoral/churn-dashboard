import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and metadata
model = joblib.load('churn_model.pkl')
features = joblib.load('model_features.pkl')
dataset = pd.read_csv('CustomerData.csv')

# Preprocess
X = dataset[features]
dataset['Churn Probability'] = model.predict_proba(X)[:, 1]
dataset['Predicted Churn'] = model.predict(X)

# Title
st.title("📊 Customer Churn Dashboard")

# Section 1: Metrics
st.subheader("📈 Model Performance Metrics")
accuracy = (dataset['Predicted Churn'] == (dataset['Churn'] == 'Yes')).mean()
st.metric("Accuracy", f"{accuracy:.2%}")

# Section 2: Churn Risk Chart
st.subheader("🔥 Top Customers at Risk of Churn")
top_risk = dataset[['customerID', 'Churn Probability']].sort_values(by='Churn Probability', ascending=False).head(10)
st.dataframe(top_risk)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top_risk, x='Churn Probability', y='customerID', palette='Reds_r', ax=ax)
ax.set_title("Top 10 At-Risk Customers")
st.pyplot(fig)

# Section 3: Feature Importance
st.subheader("💡 Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax2)
st.pyplot(fig2)

# Section 4: Upload for Prediction
st.subheader("📥 Predict New Customers")
uploaded = st.file_uploader("Upload a CSV file with customer data", type="csv")
if uploaded:
    new_data = pd.read_csv(uploaded)
    new_data = new_data.reindex(columns=features, fill_value=0)
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]
    new_data['Churn Prediction'] = predictions
    new_data['Churn Probability'] = probabilities
    st.dataframe(new_data[['Churn Prediction', 'Churn Probability'] + features])
