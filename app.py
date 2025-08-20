import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter
import re

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Dataset - 300k Registros",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar datos
@st.cache_data
def load_data():
    try:
        return pd.read_parquet('datos.parquet')
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Función para limpiar palabras
def clean_words(text_series, min_len=3, remove_numbers=True):
    """Limpia palabras eliminando números, palabras cortas y caracteres especiales"""
    all_words = []
    
    for text in text_series.dropna():
        if pd.isna(text):
            continue
        
        # Convertir a string y dividir en palabras
        words = str(text).lower().split()
        
        for word in words:
            # Limpiar caracteres especiales
            clean_word = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ]', '', word)
            
            # Aplicar filtros
            if len(clean_word) >= min_len:
                if remove_numbers and clean_word.isdigit():
                    continue
                all_words.append(clean_word)
    
    return all_words

# Función principal
def main():
    st.title("📊 Análisis del Dataset")
    st.markdown("---")
    
    # Cargar datos
    with st.spinner("Cargando dataset..."):
        df = load_data()
    
    if df is None:
        st.error("No se pudo cargar el dataset. Verifica que el archivo 'datos.parquet' existe.")
        return
    
    # Sidebar con información general
    st.sidebar.header("📈 Información General")
    st.sidebar.metric("Total de Registros", f"{len(df):,}")
    st.sidebar.metric("Total de Columnas", len(df.columns))
    
    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Explorar Dataset", 
        "📝 Análisis de Nombres", 
        "📊 Distribución de Probabilidades",
        "🔎 Filtros Avanzados"
    ])
    
    # TAB 1: Explorar Dataset
    with tab1:
        st.header("🔍 Exploración del Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Vista General de los Datos")
            # Mostrar solo las primeras 100 filas para evitar problemas de rendimiento
            st.dataframe(df.head(100), use_container_width=True)
        
        with col2:
            st.subheader("Información del Dataset")
            
            st.write("**Columnas disponibles:**")
            for col in df.columns:
                st.write(f"• {col}")
            
            st.write(f"\n**Tipos de datos:**")
            for col in df.columns[:10]:  # Solo mostrar las primeras 10
                st.text(f"{col}: {df[col].dtype}")
            
            # Valores nulos
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            if len(null_cols) > 0:
                st.write("**Valores nulos:**")
                for col in null_cols.index[:5]:  # Solo mostrar los primeros 5
                    st.text(f"{col}: {null_cols[col]:,}")
    
    # TAB 2: Análisis de Nombres
    with tab2:
        st.header("📝 Análisis de Nombres de Documentos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏷️ Top Palabras en doc_name")
            
            # Solo control para número de palabras
            top_n = st.slider("Top N palabras", 10, 50, 20)
            
            try:
                # Procesar palabras con valores por defecto
                words = clean_words(df['doc_name'], min_len=3, remove_numbers=True)
                word_counts = Counter(words)
                
                if word_counts:
                    top_words = word_counts.most_common(top_n)
                    
                    # Crear DataFrame para visualización
                    words_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
                    
                    # Gráfico de barras
                    fig = px.bar(
                        words_df, 
                        x='Frecuencia', 
                        y='Palabra',
                        orientation='h',
                        title=f"Top {top_n} Palabras más Frecuentes en doc_name"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla con detalles
                    st.dataframe(words_df, use_container_width=True)
                else:
                    st.warning("No se encontraron palabras con los criterios especificados")
                    
            except Exception as e:
                st.error(f"Error procesando palabras: {e}")
        
        with col2:
            st.subheader("📄 Top Original Names")
            
            # Análisis de original_name
            top_n_names = st.slider("Top N nombres originales", 10, 30, 15)
            
            try:
                original_counts = df['original_name'].value_counts().head(top_n_names)
                
                # Gráfico
                fig = px.bar(
                    x=original_counts.values,
                    y=original_counts.index,
                    orientation='h',
                    title=f"Top {top_n_names} Nombres Originales más Repetidos",
                    labels={'x': 'Frecuencia', 'y': 'Nombre Original'}
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla detallada
                names_df = pd.DataFrame({
                    'Nombre Original': original_counts.index,
                    'Frecuencia': original_counts.values,
                    'Porcentaje': (original_counts.values / len(df) * 100).round(2)
                })
                st.dataframe(names_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analizando nombres originales: {e}")
    
    # TAB 3: Distribución de Probabilidades (Simplificado)
    with tab3:
        st.header("📊 Análisis de Probabilidades")
        
        # Columnas de probabilidad
        prob_cols = ['lvl0_prob', 'lvl1_prob', 'lvl2_prob', 'lvl3_prob', 'p_max', 'p_second_max']
        
        # Verificar que las columnas existen
        existing_prob_cols = [col for col in prob_cols if col in df.columns]
        
        if not existing_prob_cols:
            st.error("No se encontraron columnas de probabilidad en el dataset")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Distribuciones de Probabilidades")
            
            # Selector de columna
            selected_prob = st.selectbox("Seleccionar columna de probabilidad:", existing_prob_cols)
            
            try:
                # Histograma
                fig = px.histogram(
                    df, 
                    x=selected_prob, 
                    nbins=50,
                    title=f"Distribución de {selected_prob}"
                )
                
                # Añadir línea de media
                mean_val = df[selected_prob].mean()
                fig.add_vline(
                    x=mean_val, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Media: {mean_val:.3f}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creando histograma: {e}")
        
        with col2:
            st.subheader("📊 Estadísticas")
            
            try:
                # Estadísticas de la columna seleccionada
                stats = df[selected_prob].describe()
                
                st.metric("Media", f"{stats['mean']:.4f}")
                st.metric("Mediana", f"{stats['50%']:.4f}")
                st.metric("Desv. Estándar", f"{stats['std']:.4f}")
                st.metric("Mínimo", f"{stats['min']:.4f}")
                st.metric("Máximo", f"{stats['max']:.4f}")
                
                # Mostrar distribución por cuartiles
                st.write("**Cuartiles:**")
                st.write(f"Q1 (25%): {stats['25%']:.4f}")
                st.write(f"Q2 (50%): {stats['50%']:.4f}")
                st.write(f"Q3 (75%): {stats['75%']:.4f}")
                
            except Exception as e:
                st.error(f"Error calculando estadísticas: {e}")
    
    # TAB 4: Filtros Avanzados (Corregido)
    with tab4:
        st.header("🔎 Filtros Avanzados y Exploración")
        
        # Controles de filtro en la misma página
        st.subheader("🎛️ Configurar Filtros")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Filtro por área
            if 'area' in df.columns:
                areas_unique = df['area'].dropna().unique()
                areas = ['Todas'] + sorted(areas_unique.tolist())
                selected_area = st.selectbox("Filtrar por Área:", areas)
            else:
                selected_area = 'Todas'
        
        with filter_col2:
            # Filtro por agente
            if 'agent' in df.columns:
                agents_unique = df['agent'].dropna().unique()
                agents = ['Todos'] + sorted(agents_unique.tolist())
                selected_agent = st.selectbox("Filtrar por Agente:", agents)
            else:
                selected_agent = 'Todos'
        
        with filter_col3:
            # Filtro por delta
            if 'delta' in df.columns:
                delta_min = float(df['delta'].min())
                delta_max = float(df['delta'].max())
                delta_range = st.slider(
                    "Rango de Delta:",
                    delta_min,
                    delta_max,
                    (delta_min, delta_max),
                    step=0.001
                )
            else:
                delta_range = None
        
        # Aplicar filtros
        try:
            filtered_df = df.copy()
            
            # Aplicar filtro de área
            if selected_area != 'Todas' and 'area' in df.columns:
                filtered_df = filtered_df[filtered_df['area'] == selected_area]
            
            # Aplicar filtro de agente
            if selected_agent != 'Todos' and 'agent' in df.columns:
                filtered_df = filtered_df[filtered_df['agent'] == selected_agent]
            
            # Aplicar filtro de delta
            if delta_range is not None and 'delta' in df.columns:
                filtered_df = filtered_df[
                    (filtered_df['delta'] >= delta_range[0]) & 
                    (filtered_df['delta'] <= delta_range[1])
                ]
            
            # Mostrar resultados
            st.subheader(f"📋 Datos Filtrados ({len(filtered_df):,} registros de {len(df):,} totales)")
            
            if len(filtered_df) > 0:
                # Selector de columnas a mostrar
                default_cols = ['original_name', 'doc_name', 'area', 'delta', 'p_max']
                available_cols = [col for col in default_cols if col in df.columns]
                if not available_cols:
                    available_cols = df.columns.tolist()[:5]
                
                columns_to_show = st.multiselect(
                    "Seleccionar columnas a mostrar:",
                    df.columns.tolist(),
                    default=available_cols
                )
                
                if columns_to_show:
                    # Mostrar solo las primeras 1000 filas para mejor rendimiento
                    display_df = filtered_df[columns_to_show].head(1000)
                    st.dataframe(display_df, use_container_width=True)
                    
                    if len(filtered_df) > 1000:
                        st.info(f"Mostrando las primeras 1000 filas de {len(filtered_df):,} resultados")
                
                # Estadísticas rápidas
                st.subheader("📊 Estadísticas de los Datos Filtrados")
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Registros filtrados", f"{len(filtered_df):,}")
                    st.metric("% del total", f"{len(filtered_df)/len(df)*100:.1f}%")
                
                with stats_col2:
                    if 'delta' in filtered_df.columns:
                        st.metric("Delta promedio", f"{filtered_df['delta'].mean():.4f}")
                    if 'p_max' in filtered_df.columns:
                        st.metric("p_max promedio", f"{filtered_df['p_max'].mean():.4f}")
                
                with stats_col3:
                    if 'area' in filtered_df.columns:
                        st.metric("Áreas únicas", filtered_df['area'].nunique())
                    if 'agent' in filtered_df.columns:
                        st.metric("Agentes únicos", filtered_df['agent'].nunique())
                
                # Botón para descargar (solo si hay pocos registros)
                if len(filtered_df) <= 10000:
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    if columns_to_show:
                        csv = convert_df_to_csv(filtered_df[columns_to_show])
                        st.download_button(
                            label="📥 Descargar datos filtrados como CSV",
                            data=csv,
                            file_name='datos_filtrados.csv',
                            mime='text/csv'
                        )
                else:
                    st.warning("Demasiados registros para descargar. Aplica más filtros para habilitar la descarga.")
            
            else:
                st.warning("No se encontraron registros con los filtros aplicados")
                
        except Exception as e:
            st.error(f"Error aplicando filtros: {e}")

if __name__ == "__main__":
    main()