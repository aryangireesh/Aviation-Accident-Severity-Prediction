import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import shap
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Aviation Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("events.csv", encoding="latin1", low_memory=False)
    df["event_year"] = pd.to_datetime(df["ev_date"], errors="coerce").dt.year
    return df

@st.cache_data
def load_aircraft():
    return pd.read_csv("aircraft.csv", encoding="latin1", low_memory=False)

df = load_data()
aircraft_df = load_aircraft()

# -----------------------------
# CREATE TARGET
# -----------------------------
if "severity_class" not in df.columns:
    df["severity_class"] = df["inj_tot_f"].apply(lambda x: 1 if x > 0 else 0)

# -----------------------------
# MERGE DATA
# -----------------------------
merged = df.merge(aircraft_df, on="ev_id", how="left")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("🔎 Filters")

year = st.sidebar.slider(
    "Select Year",
    int(df["event_year"].min()),
    int(df["event_year"].max()),
    2015
)

country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(df["ev_country"].dropna().unique().tolist())
)

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["All"] + sorted(df["wx_cond_basic"].dropna().unique().tolist())
)

df_filtered = df[df["event_year"] == year]

if country != "All":
    df_filtered = df_filtered[df_filtered["ev_country"] == country]

if weather != "All":
    df_filtered = df_filtered[df_filtered["wx_cond_basic"] == weather]

merged_filtered = df_filtered.merge(aircraft_df, on="ev_id", how="left")

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1 style='text-align:center;'> Aviation Accident Severity Prediction Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# KPI METRICS
# -----------------------------
total_accidents = len(df_filtered)
severe_accidents = df_filtered["severity_class"].sum()
severe_percent = (severe_accidents / total_accidents) * 100 if total_accidents > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Accidents", total_accidents)
col2.metric("Severe Accidents", int(severe_accidents))
col3.metric("Severe %", f"{severe_percent:.2f}%")

st.markdown("---")

# -----------------------------
# CHARTS ROW 1
# -----------------------------
col1, col2 = st.columns(2)

# Severity Distribution
with col1:
    st.subheader("Severity Distribution")

    severity_counts = df_filtered["severity_class"].value_counts().reset_index()
    severity_counts.columns = ["severity_class", "count"]

    fig = px.bar(
        severity_counts,
        x="severity_class",
        y="count",
        color="severity_class",
        title="Severity (0 = Non-Severe, 1 = Severe)"
    )

    st.plotly_chart(fig, use_container_width=True)

# Weather Severity
with col2:
    st.subheader("Avg Severity by Weather")

    if "wx_cond_basic" in df.columns:
        weather_avg = df.groupby("wx_cond_basic")["severity_class"].mean().reset_index()

        fig = px.bar(
            weather_avg.sort_values("severity_class", ascending=False).head(10),
            x="wx_cond_basic",
            y="severity_class",
            title="Weather Risk"
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# CHARTS ROW 2
# -----------------------------
col1, col2 = st.columns(2)

# High Risk Manufacturers
with col1:
    st.subheader("🏭 High Risk Manufacturers")

    if "acft_make" in merged_filtered.columns:
        manu_stats = merged_filtered.groupby("acft_make").agg({
            "severity_class": "mean",
            "ev_id": "count"
        }).reset_index()

        manu_stats = manu_stats[manu_stats["ev_id"] > 50]

        fig = px.bar(
            manu_stats.sort_values("severity_class", ascending=False).head(10),
            x="acft_make",
            y="severity_class"
        )

        st.plotly_chart(fig, use_container_width=True)

# Top Models
with col2:
    st.subheader("✈️ Top Aircraft Models")

    if "acft_model" in merged_filtered.columns:
        top_models = merged_filtered["acft_model"].value_counts().head(10).reset_index()
        top_models.columns = ["model", "count"]

        fig = px.bar(top_models, x="model", y="count")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# CHARTS ROW 3
# -----------------------------
col1, col2 = st.columns(2)

# Top Manufacturers
with col1:
    st.subheader("🏭 Top Manufacturers")

    top_manu = merged_filtered["acft_make"].value_counts().head(10).reset_index()
    top_manu.columns = ["manufacturer", "count"]

    fig = px.bar(top_manu, x="manufacturer", y="count")
    st.plotly_chart(fig, use_container_width=True)

# Top Countries
with col2:
    st.subheader("Top Countries")

    top_country = df_filtered["ev_country"].value_counts().head(10).reset_index()
    top_country.columns = ["country", "count"]

    fig = px.bar(top_country, x="country", y="count")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# 🤖 ML PREDICTION
# -----------------------------
# -----------------------------
# 🤖 ML PREDICTION
# -----------------------------
st.header("Accident Severity Prediction")

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

col1, col2, col3 = st.columns(3)

event_year = col1.number_input("Event Year", 2000, 2025, 2015)
wx_temp = col2.number_input("Temperature", -50, 50, 10)
acft_year = col3.number_input("Aircraft Year", 1950, 2025, 2000)

num_eng = col1.number_input("Engines", 1, 4, 1)
eng_time_total = col2.number_input("Engine Time", 0, 50000, 1000)
severity_score = col3.number_input("Severity Score", 0, 10, 5)

aircraft_age = event_year - acft_year

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🔍 Predict"):

    # Create input
    input_data = {
        "event_year": event_year,
        "wx_temp": wx_temp,
        "acft_year": acft_year,
        "num_eng": num_eng,
        "eng_time_total": eng_time_total,
        "severity_score": severity_score,
        "aircraft_age": aircraft_age
    }

    input_df = pd.DataFrame([input_data])

    # Match training columns
    for col in columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[columns]

    # Prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    result = "Severe" if pred == 1 else "Non-Severe"

    st.success(f"Prediction: {result}")
    st.info(f"Probability: {prob:.2f}")

    # -----------------------------
    # ⚠️ RISK INSIGHTS
    # -----------------------------
    st.subheader("⚠️ Risk Insights")

    if wx_temp < 0:
        st.warning("❄️ Low temperature (icing risk)")
    if eng_time_total > 2000:
        st.warning("⚙️ High engine usage")
    if num_eng == 1:
        st.warning("🛩️ Single engine risk")
    if aircraft_age > 25:
        st.warning("📆 Old aircraft")

    # -----------------------------
    # 🧠 SHAP (FINAL FIXED)
    # -----------------------------
   # -----------------------------
# -----------------------------
# 🧠 SHAP (FINAL STABLE VERSION)
# -----------------------------
st.subheader(" AI Feature Impact")

try:
    explainer = shap.TreeExplainer(model)

    # Use model internal feature importance as fallback
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

        # Match only available columns
        feature_names = columns

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        imp_df = imp_df.sort_values(by="Importance", ascending=False).head(10)

        st.write("Top Features Influencing Prediction")

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Model does not support feature importance")

except Exception as e:
    st.error(f"Explanation error: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Aviation Accident Analysis")