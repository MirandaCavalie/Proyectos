import streamlit as st
import pandas as pd

st.title("Prueba Dataset - 300k registros")

@st.cache_data
def load_data():
    return pd.read_parquet('datos.parquet')

try:
    df = load_data()
    st.success(f"✅ Dataset cargado exitosamente: {len(df):,} registros")
    st.write("Primeras filas:")
    st.dataframe(df.head())
    
    st.write("Información del dataset:")
    st.write(f"- Filas: {len(df):,}")
    st.write(f"- Columnas: {len(df.columns)}")
    
except Exception as e:
    st.error(f"Error al cargar datos: {e}")