import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    return pd.read_parquet('datos.parquet')

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
    st.title("📊 Análisis Completo del Dataset")
    st.markdown("---")
    
    # Cargar datos
    with st.spinner("Cargando dataset..."):
        df = load_data()
    
    # Sidebar con información general
    st.sidebar.header("📈 Información General")
    st.sidebar.metric("Total de Registros", f"{len(df):,}")
    st.sidebar.metric("Total de Columnas", len(df.columns))
    st.sidebar.metric("Memoria Utilizada", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Crear pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Explorar Dataset", 
        "📝 Análisis de Nombres", 
        "📊 Distribución de Probabilidades",
        "📈 Análisis Delta",
        "🔎 Filtros Avanzados"
    ])
    
    # TAB 1: Explorar Dataset
    with tab1:
        st.header("🔍 Exploración del Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Vista General de los Datos")
            st.dataframe(df.head(1000), use_container_width=True)
        
        with col2:
            st.subheader("Estadísticas Básicas")
            
            # Información por columna
            st.write("**Tipos de datos:**")
            for col in df.columns:
                st.text(f"{col}: {df[col].dtype}")
            
            st.write("**Valores nulos:**")
            null_counts = df.isnull().sum()
            for col in df.columns:
                if null_counts[col] > 0:
                    st.text(f"{col}: {null_counts[col]:,}")
        
        # Resumen estadístico de columnas numéricas
        st.subheader("📊 Resumen Estadístico")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe())
    
    # TAB 2: Análisis de Nombres
    with tab2:
        st.header("📝 Análisis de Nombres de Documentos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏷️ Top Palabras en doc_name")
            
            # Controles para filtrado
            min_length = st.slider("Longitud mínima de palabra", 2, 8, 3)
            remove_nums = st.checkbox("Eliminar números", True)
            top_n = st.slider("Top N palabras", 10, 100, 20)
            
            # Procesar palabras
            words = clean_words(df['doc_name'], min_len=min_length, remove_numbers=remove_nums)
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
                    title=f"Top {top_n} Palabras más Frecuentes en doc_name",
                    color='Frecuencia',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla con detalles
                st.dataframe(words_df, use_container_width=True)
            else:
                st.warning("No se encontraron palabras con los criterios especificados")
        
        with col2:
            st.subheader("📄 Top Original Names")
            
            # Análisis de original_name
            top_n_names = st.slider("Top N nombres originales", 10, 50, 15, key="names")
            
            original_counts = df['original_name'].value_counts().head(top_n_names)
            
            # Gráfico
            fig = px.bar(
                x=original_counts.values,
                y=original_counts.index,
                orientation='h',
                title=f"Top {top_n_names} Nombres Originales más Repetidos",
                labels={'x': 'Frecuencia', 'y': 'Nombre Original'},
                color=original_counts.values,
                color_continuous_scale='plasma'
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
    
    # TAB 3: Distribución de Probabilidades
    with tab3:
        st.header("📊 Análisis de Distribución de Probabilidades")
        
        # Columnas de probabilidad
        prob_cols = ['lvl0_prob', 'lvl1_prob', 'lvl2_prob', 'lvl3_prob', 'p_max', 'p_second_max']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribuciones Individuales")
            
            # Selector de columna
            selected_prob = st.selectbox("Seleccionar columna de probabilidad:", prob_cols)
            
            # Histograma
            fig = px.histogram(
                df, 
                x=selected_prob, 
                nbins=50,
                title=f"Distribución de {selected_prob}",
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(
                x=df[selected_prob].mean(), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Media: {df[selected_prob].mean():.3f}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas
            st.write(f"**Estadísticas de {selected_prob}:**")
            stats = df[selected_prob].describe()
            for stat, value in stats.items():
                st.metric(stat.capitalize(), f"{value:.4f}")
        
        with col2:
            st.subheader("📊 Comparación de Probabilidades")
            
            # Box plots comparativos
            fig = go.Figure()
            
            for col in prob_cols:
                fig.add_trace(go.Box(
                    y=df[col].dropna(),
                    name=col,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title="Distribución Comparativa de Probabilidades",
                yaxis_title="Valor de Probabilidad",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de correlaciones
        st.subheader("🔥 Matriz de Correlación entre Probabilidades")
        corr_matrix = df[prob_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlación entre Variables de Probabilidad"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Análisis Delta
    with tab4:
        st.header("📈 Análisis de la Variable Delta")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribución de Delta")
            
            # Histograma de delta
            fig = px.histogram(
                df, 
                x='delta', 
                nbins=50,
                title="Distribución de Delta",
                marginal="box"  # Añade box plot arriba
            )
            
            # Añadir líneas de estadísticas
            mean_delta = df['delta'].mean()
            median_delta = df['delta'].median()
            
            fig.add_vline(x=mean_delta, line_dash="dash", line_color="red", 
                         annotation_text=f"Media: {mean_delta:.3f}")
            fig.add_vline(x=median_delta, line_dash="dash", line_color="green", 
                         annotation_text=f"Mediana: {median_delta:.3f}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas detalladas
            st.write("**Estadísticas de Delta:**")
            delta_stats = df['delta'].describe()
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Media", f"{delta_stats['mean']:.4f}")
                st.metric("Mediana", f"{delta_stats['50%']:.4f}")
                st.metric("Desv. Estándar", f"{delta_stats['std']:.4f}")
                st.metric("Mínimo", f"{delta_stats['min']:.4f}")
            
            with metrics_col2:
                st.metric("Máximo", f"{delta_stats['max']:.4f}")
                st.metric("Q1", f"{delta_stats['25%']:.4f}")
                st.metric("Q3", f"{delta_stats['75%']:.4f}")
                st.metric("Rango", f"{delta_stats['max'] - delta_stats['min']:.4f}")
        
        with col2:
            st.subheader("🎯 Análisis por Rangos")
            
            # Crear rangos de delta
            df['delta_range'] = pd.cut(df['delta'], bins=5, precision=3)
            range_counts = df['delta_range'].value_counts().sort_index()
            
            # Gráfico de barras por rangos
            fig = px.bar(
                x=range_counts.index.astype(str),
                y=range_counts.values,
                title="Distribución por Rangos de Delta",
                labels={'x': 'Rango de Delta', 'y': 'Cantidad'},
                color=range_counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de rangos
            range_df = pd.DataFrame({
                'Rango': range_counts.index.astype(str),
                'Cantidad': range_counts.values,
                'Porcentaje': (range_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(range_df, use_container_width=True)
        
        # Scatter plot: Delta vs otras variables
        st.subheader("🔗 Relación Delta con otras Variables")
        
        scatter_col = st.selectbox(
            "Seleccionar variable para comparar con Delta:",
            ['p_max', 'p_second_max', 'lvl0_prob', 'lvl1_prob', 'lvl2_prob', 'lvl3_prob']
        )
        
        fig = px.scatter(
            df.sample(5000),  # Muestra para mejor rendimiento
            x='delta', 
            y=scatter_col,
            title=f"Relación entre Delta y {scatter_col}",
            alpha=0.6,
            color='delta',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: Filtros Avanzados
    with tab5:
        st.header("🔎 Filtros Avanzados y Exploración")
        
        # Sidebar para filtros
        with st.sidebar:
            st.header("🎛️ Filtros")
            
            # Filtro por área
            if 'area' in df.columns:
                areas = ['Todas'] + sorted(df['area'].dropna().unique().tolist())
                selected_area = st.selectbox("Filtrar por Área:", areas)
            
            # Filtro por agente
            if 'agent' in df.columns:
                agents = ['Todos'] + sorted(df['agent'].dropna().unique().tolist())
                selected_agent = st.selectbox("Filtrar por Agente:", agents)
            
            # Filtro por delta
            delta_range = st.slider(
                "Rango de Delta:",
                float(df['delta'].min()),
                float(df['delta'].max()),
                (float(df['delta'].min()), float(df['delta'].max())),
                step=0.001
            )
            
            # Filtro por p_max
            pmax_range = st.slider(
                "Rango de p_max:",
                float(df['p_max'].min()),
                float(df['p_max'].max()),
                (float(df['p_max'].min()), float(df['p_max'].max())),
                step=0.01
            )
        
        # Aplicar filtros
        filtered_df = df.copy()
        
        if 'selected_area' in locals() and selected_area != 'Todas':
            filtered_df = filtered_df[filtered_df['area'] == selected_area]
        
        if 'selected_agent' in locals() and selected_agent != 'Todos':
            filtered_df = filtered_df[filtered_df['agent'] == selected_agent]
        
        filtered_df = filtered_df[
            (filtered_df['delta'] >= delta_range[0]) & 
            (filtered_df['delta'] <= delta_range[1])
        ]
        
        filtered_df = filtered_df[
            (filtered_df['p_max'] >= pmax_range[0]) & 
            (filtered_df['p_max'] <= pmax_range[1])
        ]
        
        # Mostrar resultados filtrados
        st.subheader(f"📋 Datos Filtrados ({len(filtered_df):,} registros)")
        
        if len(filtered_df) > 0:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Selector de columnas a mostrar
                columns_to_show = st.multiselect(
                    "Seleccionar columnas a mostrar:",
                    df.columns.tolist(),
                    default=['original_name', 'doc_name', 'area', 'delta', 'p_max', 'agent']
                )
                
                if columns_to_show:
                    st.dataframe(filtered_df[columns_to_show], use_container_width=True)
                
                # Botón para descargar datos filtrados
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = convert_df_to_csv(filtered_df[columns_to_show] if columns_to_show else filtered_df)
                st.download_button(
                    label="📥 Descargar datos filtrados como CSV",
                    data=csv,
                    file_name='datos_filtrados.csv',
                    mime='text/csv'
                )
            
            with col2:
                st.subheader("📊 Estadísticas de Filtrado")
                st.metric("Registros filtrados", f"{len(filtered_df):,}")
                st.metric("% del total", f"{len(filtered_df)/len(df)*100:.1f}%")
                
                if len(filtered_df) > 0:
                    st.write("**Estadísticas rápidas:**")
                    st.write(f"Delta promedio: {filtered_df['delta'].mean():.3f}")
                    st.write(f"p_max promedio: {filtered_df['p_max'].mean():.3f}")
                    
                    if 'area' in filtered_df.columns:
                        st.write(f"Áreas únicas: {filtered_df['area'].nunique()}")
                    
                    if 'agent' in filtered_df.columns:
                        st.write(f"Agentes únicos: {filtered_df['agent'].nunique()}")
        else:
            st.warning("No se encontraron registros con los filtros aplicados")

if __name__ == "__main__":
    main()