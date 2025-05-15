import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="DLP Policy Report", layout="wide")

st.title("📊 DLP Policy Report – Windows + OneDrive")

# Cargar datos
doc_dist = pd.read_csv("data/document_distribution.csv")
env_comp = pd.read_csv("data/environment_comparison.csv")
strict = pd.read_csv("data/strict_enforcement.csv")
monitor = pd.read_csv("data/monitoring_alerting.csv")
resumen = pd.read_csv("data/resumen_general.csv")

# Pie chart
st.subheader("Document Distribution")
fig = go.Figure(data=[go.Pie(
    labels=doc_dist["name"],
    values=doc_dist["percentage"],
    marker=dict(colors=doc_dist["color"]),
    hole=0.4
)])
st.plotly_chart(fig, use_container_width=True)

# Bar chart
st.subheader("Environment Comparison")
fig2 = go.Figure()
fig2.add_trace(go.Bar(name="Total Docs", x=env_comp["name"], y=env_comp["documents"]))
fig2.add_trace(go.Bar(name="Confidential Docs", x=env_comp["name"], y=env_comp["confidentialDocs"]))
fig2.add_trace(go.Bar(name="Strict Enforcement", x=env_comp["name"], y=env_comp["strictEnforcement"]))
fig2.add_trace(go.Bar(name="Monitoring & Alerting", x=env_comp["name"], y=env_comp["monitoringAlerting"]))
fig2.update_layout(barmode='group')
st.plotly_chart(fig2, use_container_width=True)

# Tabla resumen
st.subheader("Executive Summary")
st.dataframe(resumen)

# Strict Enforcement Breakdown
st.subheader("Strict Enforcement")
st.dataframe(strict)

# Monitoring and Alerting
st.subheader("Monitoring and Alerting")
st.dataframe(monitor)

st.markdown("---")
st.markdown("© 2025 Kriptos - DLP Policy Report | support@kriptos.io")
