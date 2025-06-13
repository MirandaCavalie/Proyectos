import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import re
import os

# Configuración de la página
st.set_page_config(
    page_title="📊 Dashboard de Templates",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de credenciales
CREDENTIALS_PATH = r"C:\Users\M.CavalieG\OneDrive - Universidad del Pacífico\Documents\Kriptos\Trabajo\Proyectos\kriptos-4624e10fab77.json"

# Configuración de BigQuery
PROJECT_ID = "kriptos"
DATASET_ID = "kriptos_data"
TABLE_ID = "template_data"

# Configurar credenciales si el archivo existe
if os.path.exists(CREDENTIALS_PATH):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

# Importar BigQuery
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError as e:
    BIGQUERY_AVAILABLE = False

@st.cache_data(ttl=600)  # Cache por 10 minutos
def load_data_from_bigquery():
    """
    Carga datos desde BigQuery con credenciales configuradas
    """
    if not BIGQUERY_AVAILABLE:
        return None, "BigQuery no está instalado"
    
    if not os.path.exists(CREDENTIALS_PATH):
        return None, f"Archivo de credenciales no encontrado: {CREDENTIALS_PATH}"
    
    try:
        # Configurar credenciales explícitamente
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
        
        client = bigquery.Client(project=PROJECT_ID)
        
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        LIMIT 1000000
        """
        
        df = client.query(query).to_dataframe()
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def process_data(df):
    """
    Procesa y limpia los datos
    """
    if df is None:
        return None, []
    
    # Convertir fecha de forma segura - manejo de formato ISO con timezone
    if 'modif_date' in df.columns:
        try:
            # Intentar convertir a datetime - pandas maneja automáticamente ISO format con timezone
            df['modif_date'] = pd.to_datetime(df['modif_date'], errors='coerce', utc=True)
            
            # Convertir a timezone local (opcional) y luego remover timezone para facilitar procesamiento
            df['modif_date'] = df['modif_date'].dt.tz_convert(None)
            
            # Solo crear columnas derivadas si la conversión fue exitosa
            if df['modif_date'].notna().any():
                df['year_month'] = df['modif_date'].dt.to_period('M').astype(str)
                df['year'] = df['modif_date'].dt.year
                df['month'] = df['modif_date'].dt.month
                df['date_only'] = df['modif_date'].dt.date
                st.success(f"✅ Fechas procesadas correctamente: {df['modif_date'].notna().sum():,} fechas válidas")
            else:
                st.warning("⚠️ No se pudieron procesar las fechas en modif_date")
        except Exception as e:
            st.warning(f"⚠️ Error procesando fechas: {str(e)}")
    
    # Identificar columnas de contadores de forma segura
    counter_columns = []
    for col in df.columns:
        if 'counter' in str(col).lower():
            counter_columns.append(col)
    
    # Limpiar doc_name
    if 'doc_name' in df.columns:
        df['doc_name_clean'] = df['doc_name'].fillna('Sin nombre')
    
    # Limpiar classifications
    if 'classifications' in df.columns:
        df['classifications_clean'] = df['classifications'].fillna('Sin clasificación')
    
    return df, counter_columns

def create_summary_metrics(df):
    """
    Crea métricas de resumen de forma segura
    """
    total_templates = len(df)
    unique_classifications = df['classifications_clean'].nunique() if 'classifications_clean' in df.columns else 0
    date_range = ""
    
    # Manejo seguro de fechas
    try:
        if 'modif_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['modif_date']):
            valid_dates = df['modif_date'].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                date_range = f"{min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
    except Exception as e:
        st.warning(f"⚠️ Error procesando rango de fechas: {str(e)}")
    
    return total_templates, unique_classifications, date_range

def plot_classification_distribution(df):
    """
    Gráfico de distribución por clasificación
    """
    if 'classifications_clean' not in df.columns:
        return None
    
    class_counts = df['classifications_clean'].value_counts().head(15)
    
    fig = px.bar(
        x=class_counts.values,
        y=class_counts.index,
        orientation='h',
        title="📋 Distribución de Templates por Clasificación",
        labels={'x': 'Cantidad de Templates', 'y': 'Clasificación'},
        color=class_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_counter_analysis(df, counter_columns):
    """
    Análisis de contadores
    """
    if not counter_columns:
        return None
    
    # Calcular estadísticas de contadores
    counter_stats = []
    
    for col in counter_columns:
        if col in df.columns:
            non_zero_count = (df[col] > 0).sum() if df[col].dtype in ['int64', 'float64'] else 0
            total_count = len(df)
            percentage = (non_zero_count / total_count) * 100 if total_count > 0 else 0
            
            counter_stats.append({
                'Counter': col.replace('_', ' ').title(),
                'Documentos con Datos': non_zero_count,
                'Porcentaje': percentage,
                'Total': total_count
            })
    
    if not counter_stats:
        return None
    
    counter_df = pd.DataFrame(counter_stats)
    counter_df = counter_df.sort_values('Porcentaje', ascending=True)
    
    fig = px.bar(
        counter_df,
        x='Porcentaje',
        y='Counter',
        orientation='h',
        title="🔢 Análisis de Contadores - % de Documentos con Datos",
        labels={'Porcentaje': '% de Documentos', 'Counter': 'Tipo de Contador'},
        color='Porcentaje',
        color_continuous_scale='plasma',
        text='Documentos con Datos'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(height=max(400, len(counter_stats) * 30))
    
    return fig

def plot_temporal_analysis(df):
    """
    Análisis temporal con manejo robusto de errores
    """
    if 'modif_date' not in df.columns:
        return None
    
    try:
        # Verificar si modif_date es datetime
        if not pd.api.types.is_datetime64_any_dtype(df['modif_date']):
            return None
            
        if df['modif_date'].isna().all():
            return None
        
        # Filtrar fechas válidas
        df_temporal = df[df['modif_date'].notna()].copy()
        
        if len(df_temporal) == 0:
            return None
        
        # Verificar si tenemos year_month
        if 'year_month' not in df_temporal.columns:
            df_temporal['year_month'] = df_temporal['modif_date'].dt.to_period('M').astype(str)
        
        # Agrupar por mes
        monthly_counts = df_temporal.groupby('year_month').size().reset_index(name='count')
        monthly_counts['year_month'] = pd.to_datetime(monthly_counts['year_month'])
        monthly_counts = monthly_counts.sort_values('year_month')
        
        fig = px.line(
            monthly_counts,
            x='year_month',
            y='count',
            title="📈 Evolución Temporal de Templates",
            labels={'year_month': 'Fecha', 'count': 'Cantidad de Templates'},
            markers=True
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Fecha",
            yaxis_title="Cantidad de Templates"
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"⚠️ Error en análisis temporal: {str(e)}")
        return None

def analyze_document_names(df, is_template_filter=None):
    """
    Análisis de nombres de documentos - simplificado porque toda la data son templates
    """
    if 'doc_name_clean' not in df.columns:
        return None
    
    # Ya no hay filtros, solo mostrar los nombres más comunes
    name_counts = df['doc_name_clean'].value_counts().head(20)
    
    fig = px.bar(
        x=name_counts.values,
        y=name_counts.index,
        orientation='h',
        title="📄 Nombres de Documentos Más Comunes",
        labels={'x': 'Frecuencia', 'y': 'Nombre del Documento'},
        color=name_counts.values,
        color_continuous_scale='blues'
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_counter_heatmap(df, counter_columns):
    """
    Mapa de calor de contadores por clasificación
    """
    if not counter_columns or 'classifications_clean' not in df.columns:
        return None
    
    # Seleccionar top clasificaciones y contadores con datos
    top_classifications = df['classifications_clean'].value_counts().head(10).index
    df_filtered = df[df['classifications_clean'].isin(top_classifications)]
    
    # Crear matriz de correlación
    heatmap_data = []
    for classification in top_classifications:
        class_data = df_filtered[df_filtered['classifications_clean'] == classification]
        row_data = {'Classification': classification}
        
        for counter in counter_columns[:10]:  # Limitar a 10 contadores
            if counter in class_data.columns:
                avg_value = class_data[counter].mean() if class_data[counter].dtype in ['int64', 'float64'] else 0
                row_data[counter.replace('_', ' ').title()] = avg_value
        
        heatmap_data.append(row_data)
    
    if not heatmap_data:
        return None
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('Classification')
    
    fig = px.imshow(
        heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        title="🔥 Mapa de Calor: Contadores por Clasificación",
        color_continuous_scale='viridis',
        aspect='auto'
    )
    
    fig.update_layout(height=500)
    
    return fig

# APLICACIÓN PRINCIPAL
def main():
    st.title("📊 Dashboard de Análisis de Templates")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🔧 Configuración")
    st.sidebar.markdown("### Conexión a BigQuery")
    st.sidebar.info(f"**Proyecto:** {PROJECT_ID}\n**Dataset:** {DATASET_ID}\n**Tabla:** {TABLE_ID}")
    
    # Verificar configuración
    if os.path.exists(CREDENTIALS_PATH):
        st.sidebar.success("✅ Credenciales configuradas")
    else:
        st.sidebar.error("❌ Archivo de credenciales no encontrado")
        st.sidebar.stop()
    
    if BIGQUERY_AVAILABLE:
        st.sidebar.success("✅ BigQuery disponible")
    else:
        st.sidebar.error("❌ BigQuery no disponible")
        st.sidebar.stop()
    
    # Botón para cargar datos
    if st.sidebar.button("🔄 Cargar datos desde BigQuery"):
        with st.spinner("🔄 Cargando datos desde BigQuery..."):
            df, error = load_data_from_bigquery()
            
            if error:
                st.error(f"Error conectando a BigQuery: {error}")
                st.stop()
            else:
                st.success(f"✅ Datos cargados: {len(df):,} filas")
                
                # Procesar datos
                df, counter_columns = process_data(df)
                
                # Guardar en session state
                st.session_state['df'] = df
                st.session_state['counter_columns'] = counter_columns
    
    # Verificar si hay datos en session state
    if 'df' not in st.session_state:
        st.info("👆 Haz clic en 'Cargar datos desde BigQuery' para comenzar")
        st.stop()
    
    # Obtener datos de session state
    df = st.session_state['df']
    counter_columns = st.session_state['counter_columns']
    
    # Métricas principales
    st.header("📈 Métricas Principales")
    total_templates, unique_classifications, date_range = create_summary_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Templates", f"{total_templates:,}")
    
    with col2:
        st.metric("🏷️ Clasificaciones Únicas", unique_classifications)
    
    with col3:
        st.metric("🔢 Contadores Detectados", len(counter_columns))
    
    with col4:
        if date_range:
            st.metric("📅 Rango de Fechas", date_range)
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Análisis General", 
        "🔢 Contadores", 
        "📈 Análisis Temporal", 
        "📄 Nombres de Documentos", 
        "🔍 Exploración de Datos"
    ])
    
    with tab1:
        st.header("📋 Distribución por Clasificación")
        
        fig_class = plot_classification_distribution(df)
        if fig_class:
            st.plotly_chart(fig_class, use_container_width=True)
        else:
            st.warning("No se pudo generar el gráfico de clasificaciones")
        
        # Análisis detallado por clasificación (1, 2, 3)
        if 'classifications_clean' in df.columns:
            st.header("🔍 Análisis Detallado por Clasificación")
            
            # Crear métricas por clasificación
            classifications = df['classifications_clean'].unique()
            
            # Mostrar métricas para cada clasificación
            for classification in sorted(classifications):
                if str(classification) in ['1', '2', '3']:
                    st.subheader(f"📊 Clasificación {classification}")
                    
                    class_data = df[df['classifications_clean'] == classification]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(f"📄 Templates Clase {classification}", f"{len(class_data):,}")
                    
                    with col2:
                        percentage = (len(class_data) / len(df)) * 100
                        st.metric("📊 Porcentaje del Total", f"{percentage:.1f}%")
                    
                    with col3:
                        # Promedio de contadores para esta clasificación
                        counter_cols = [col for col in class_data.columns if 'counter' in col.lower() or '_count' in col.lower()]
                        if counter_cols:
                            avg_counters = sum(class_data[col].sum() for col in counter_cols if class_data[col].dtype in ['int64', 'float64'])
                        else:
                            avg_counters = 0
                        st.metric("🔢 Total Contadores", f"{avg_counters:,}")
                    
                    with col4:
                        # Documentos únicos por nombre
                        unique_docs = class_data['doc_name_clean'].nunique() if 'doc_name_clean' in class_data.columns else 0
                        st.metric("📝 Nombres Únicos", f"{unique_docs:,}")
                    
                    st.markdown("---")
    
    with tab2:
        st.header("🔢 Análisis de Contadores")
        
        # Análisis general primero
        if counter_columns:
            fig_counters = plot_counter_analysis(df, counter_columns)
            if fig_counters:
                st.plotly_chart(fig_counters, use_container_width=True)
            
            # Análisis detallado de contadores específicos
            st.header("🎯 Análisis Detallado de Contadores")
            
            # Definir las columnas específicas de contadores
            specific_counters = {
                'Tarjetas de Crédito': [
                    'cc_discover_count', 'cc_visa_count', 'cc_diners_club_count', 'cc_mastercard_count'
                ],
                'Información Personal (PII)': [
                    'pii_address_count', 'pii_ruc_ecu_count', 'pii_ced_ecu_count', 
                    'pii_phone_number_count', 'pii_personal_name_count', 'pii_email_count'
                ]
            }
            
            for category, counters in specific_counters.items():
                st.subheader(f"💳 {category}")
                
                # Verificar qué columnas existen
                existing_counters = [col for col in counters if col in df.columns]
                
                if existing_counters:
                    # Crear métricas para cada contador
                    cols = st.columns(len(existing_counters))
                    
                    for i, counter in enumerate(existing_counters):
                        with cols[i]:
                            if df[counter].dtype in ['int64', 'float64']:
                                total_count = df[counter].sum()
                                docs_with_data = (df[counter] > 0).sum()
                                percentage = (docs_with_data / len(df)) * 100
                                
                                # Nombre limpio para mostrar
                                clean_name = counter.replace('_count', '').replace('_', ' ').title()
                                
                                st.metric(
                                    f"🔢 {clean_name}",
                                    f"{total_count:,}",
                                    f"{docs_with_data:,} docs ({percentage:.1f}%)"
                                )
                    
                    # Gráfico específico para esta categoría
                    category_stats = []
                    for counter in existing_counters:
                        if df[counter].dtype in ['int64', 'float64']:
                            docs_with_data = (df[counter] > 0).sum()
                            percentage = (docs_with_data / len(df)) * 100
                            total_count = df[counter].sum()
                            
                            category_stats.append({
                                'Contador': counter.replace('_count', '').replace('_', ' ').title(),
                                'Total Ocurrencias': total_count,
                                'Documentos con Datos': docs_with_data,
                                'Porcentaje': percentage
                            })
                    
                    if category_stats:
                        category_df = pd.DataFrame(category_stats)
                        
                        # Gráfico de barras para esta categoría
                        fig_category = px.bar(
                            category_df,
                            x='Porcentaje',
                            y='Contador',
                            orientation='h',
                            title=f"📊 {category} - % de Documentos con Datos",
                            labels={'Porcentaje': '% de Documentos', 'Contador': 'Tipo de Contador'},
                            color='Total Ocurrencias',
                            color_continuous_scale='viridis'
                        )
                        fig_category.update_layout(height=300)
                        st.plotly_chart(fig_category, use_container_width=True)
                        
                        # Tabla detallada
                        st.dataframe(category_df, use_container_width=True)
                else:
                    st.warning(f"No se encontraron columnas para {category}")
                
                st.markdown("---")
            
            # Tabla de estadísticas generales
            st.subheader("📊 Estadísticas Generales de Todos los Contadores")
            
            counter_stats = []
            for col in counter_columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    stats = {
                        'Contador': col,
                        'Documentos con Datos': (df[col] > 0).sum(),
                        'Promedio': round(df[col].mean(), 2),
                        'Máximo': df[col].max(),
                        'Suma Total': df[col].sum()
                    }
                    counter_stats.append(stats)
            
            if counter_stats:
                counter_stats_df = pd.DataFrame(counter_stats)
                st.dataframe(counter_stats_df, use_container_width=True)
        else:
            st.warning("No se encontraron columnas de contadores")
    
    with tab3:
        st.header("📈 Análisis Temporal")
        
        fig_temporal = plot_temporal_analysis(df)
        if fig_temporal:
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Estadísticas temporales adicionales
            if 'modif_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['modif_date']):
                df_temporal = df[df['modif_date'].notna()]
                
                if len(df_temporal) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📅 Templates por Año")
                        if 'year' in df_temporal.columns:
                            yearly_counts = df_temporal.groupby('year').size()
                            st.bar_chart(yearly_counts)
                    
                    with col2:
                        st.subheader("📅 Templates por Mes")
                        if 'month' in df_temporal.columns:
                            monthly_counts = df_temporal.groupby('month').size()
                            st.bar_chart(monthly_counts)
        else:
            st.warning("No hay datos temporales suficientes para el análisis")
    
    with tab4:
        st.header("📄 Análisis de Nombres de Documentos")
        
        # Ya no hay filtro porque toda la data son templates
        st.info("💡 Todos los documentos en esta base de datos son templates")
        
        fig_names = analyze_document_names(df, None)  # Sin filtro
        if fig_names:
            st.plotly_chart(fig_names, use_container_width=True)
        
        # Estadísticas de nombres de documentos
        if 'doc_name_clean' in df.columns:
            st.subheader("📊 Estadísticas de Nombres de Documentos")
            
            # Análisis de extensiones de archivos
            if df['doc_name_clean'].notna().any():
                try:
                    extensions = df['doc_name_clean'].str.extract(r'\.([^.]+)$')[0].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📄 Total Documentos", len(df))
                    
                    with col2:
                        st.metric("📄 Nombres Únicos", df['doc_name_clean'].nunique())
                    
                    with col3:
                        if len(extensions) > 0:
                            st.metric("📎 Extensión Más Común", f"{extensions.index[0]} ({extensions.iloc[0]})")
                    
                    # Gráfico de extensiones de archivos
                    if len(extensions) > 0:
                        st.subheader("📎 Distribución por Tipo de Archivo")
                        fig_ext = px.pie(
                            values=extensions.values,
                            names=extensions.index,
                            title="Distribución de Extensiones de Archivos"
                        )
                        st.plotly_chart(fig_ext, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Error analizando extensiones: {str(e)}")
    
    with tab5:
        st.header("🔍 Exploración de Datos")
        
        # Información general de la tabla
        st.subheader("📊 Información General")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Dimensiones:** {df.shape[0]:,} filas × {df.shape[1]} columnas
            **Memoria:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            **Tipos de datos:** {df.dtypes.value_counts().to_dict()}
            """)
        
        with col2:
            st.info(f"""
            **Valores nulos por columna:**
            {df.isnull().sum().sum():,} valores nulos en total
            **Columnas con más nulos:** {df.isnull().sum().nlargest(3).to_dict()}
            """)
        
        # Filtros interactivos
        st.subheader("🔍 Filtros Interactivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'classifications_clean' in df.columns:
                selected_classifications = st.multiselect(
                    "Filtrar por Clasificación:",
                    options=df['classifications_clean'].unique(),
                    default=df['classifications_clean'].unique()[:5]
                )
            else:
                selected_classifications = []
        
        with col2:
            # Filtro de fecha
            if 'modif_date' in df.columns and df['modif_date'].notna().any():
                date_range_filter = st.date_input(
                    "Rango de fechas:",
                    value=(df['modif_date'].min().date(), df['modif_date'].max().date()),
                    min_value=df['modif_date'].min().date(),
                    max_value=df['modif_date'].max().date()
                )
        
        # Aplicar filtros
        df_filtered = df.copy()
        
        if selected_classifications and 'classifications_clean' in df.columns:
            df_filtered = df_filtered[df_filtered['classifications_clean'].isin(selected_classifications)]
        
        # Mostrar datos filtrados
        st.subheader("📋 Datos Filtrados")
        st.dataframe(df_filtered.head(1000), use_container_width=True)
        
        # Opción de descarga
        if st.button("📥 Descargar datos filtrados como CSV"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f"templates_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard creado con Streamlit y BigQuery*")

if __name__ == "__main__":
    main()