import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Dashboard")

# Load assets
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("CustomerData.csv")

@st.cache_data
def load_features():
    return joblib.load("model_features.pkl")

try:
    model = load_model()
    dataset = load_data()
    features = load_features()

    # Validate columns
    missing_cols = [col for col in features if col not in dataset.columns]
    if missing_cols:
        st.error(f"❌ Missing columns in dataset: {missing_cols}")
        st.stop()

    # Preprocessing and predictions
    X = dataset[features]
    dataset['Churn Probability'] = model.predict_proba(X)[:, 1]
    dataset['Predicted Churn'] = model.predict(X)

    # ✅ Section 1: Model Accuracy
    st.subheader("📈 Model Performance Metrics")
    if 'Churn' in dataset.columns:
        true_labels = dataset['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        accuracy = (dataset['Predicted Churn'] == true_labels).mean()
        st.metric("Accuracy", f"{accuracy:.2%}")
    else:
        st.warning("⚠️ 'Churn' column not found — skipping accuracy metric.")

    # ✅ Section 2: Top Churn Risks
    st.subheader("🔥 Top Customers at Risk of Churn")
    top_risk = dataset[['customerID', 'Churn Probability']].sort_values(
        by='Churn Probability', ascending=False).head(10)
    st.dataframe(top_risk)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_risk, x='Churn Probability', y='customerID', hue='Churn Probability',
                dodge=False, palette='Reds_r', legend=False, ax=ax)
    ax.set_title("Top 10 At-Risk Customers")
    st.pyplot(fig)

    # ✅ Section 3: Feature Importance
    st.subheader("💡 Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig2, ax2 = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax2)
        ax2.set_title("Top 10 Important Features")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ Feature importances not available for this model.")

    # ✅ Section 4: Upload for Predictions
    st.subheader("📥 Predict New Customers")
    uploaded = st.file_uploader("Upload a CSV file with customer data", type="csv")

    if uploaded:
        try:
            new_data = pd.read_csv(uploaded)
            new_data = new_data.reindex(columns=features, fill_value=0)
            new_data['Churn Prediction'] = model.predict(new_data)
            new_data['Churn Probability'] = model.predict_proba(new_data)[:, 1]
            st.dataframe(new_data[['Churn Prediction', 'Churn Probability'] + features])
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

except FileNotFoundError as e:
    st.error("❌ Required file not found. Ensure churn_model.pkl, model_features.pkl, and CustomerData.csv are in the same directory.")
except Exception as e:
    st.error(f"⚠️ Unexpected error: {e}")
