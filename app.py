import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer Churn Dashboard", layout="centered")
st.title("📊 Customer Churn Dashboard")

# Try loading data
try:
    df = pd.read_csv("CustomerData.csv")
    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("🚫 Could not find `CustomerData.csv`. Please make sure it's in your GitHub repo.")

st.markdown("---")
st.markdown("This dashboard will show churn predictions and high-risk customers once data is loaded.")
