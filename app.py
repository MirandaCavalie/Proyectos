import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Configuración de la página
st.set_page_config(
    page_title="📊 Dashboard de Templates",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de BigQuery para datos en tiempo real
PROJECT_ID = "kriptos"
DATASET_ID = "kriptos_data"
TABLE_ID = "template_data"

# Colores fijos para clasificaciones
CLASSIFICATION_COLORS = {
    '1.0': '#FFD700',  # Amarillo
    '2.0': '#FF8C00',  # Naranja
    '3.0': '#FF4500'   # Rojo
}

# Función para cargar las métricas desde el archivo JSON
@st.cache_data
def load_metrics():
    """Carga los datos de métricas desde el archivo JSON"""
    json_file = "dashboard_metrics/dashboard_metrics.json"
    
    if not os.path.exists(json_file):
        st.error(f"❌ No se encontró el archivo: {json_file}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        return None

def setup_bigquery_credentials():
    """Configura las credenciales de BigQuery de forma segura"""
    try:
        # Opción 1: Usar Streamlit Secrets
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            return credentials, None
        
        # Opción 2: Variables de entorno
        elif all(key in os.environ for key in ['GCP_PROJECT_ID', 'GCP_PRIVATE_KEY', 'GCP_CLIENT_EMAIL']):
            from google.oauth2 import service_account
            
            credentials_info = {
                "type": os.getenv("GCP_TYPE", "service_account"),
                "project_id": os.getenv("GCP_PROJECT_ID"),
                "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
                "private_key": os.getenv("GCP_PRIVATE_KEY").replace('\\n', '\n'),
                "client_email": os.getenv("GCP_CLIENT_EMAIL"),
                "client_id": os.getenv("GCP_CLIENT_ID"),
                "auth_uri": os.getenv("GCP_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.getenv("GCP_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
                "client_x509_cert_url": os.getenv("GCP_CLIENT_X509_CERT_URL")
            }
            
            credentials_info = {k: v for k, v in credentials_info.items() if v is not None}
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            return credentials, None
        
        # Opción 3: Archivo local
        else:
            local_credentials_path = "./kriptos-credentials.json"
            if os.path.exists(local_credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_credentials_path
                return None, None
            else:
                return None, "No se encontraron credenciales configuradas."
    
    except Exception as e:
        return None, f"Error configurando credenciales: {str(e)}"

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_sample_data_from_bigquery():
    """Carga una muestra de datos desde BigQuery para exploración"""
    try:
        from google.cloud import bigquery
        
        credentials, error = setup_bigquery_credentials()
        if error:
            return None, error
        
        if credentials:
            client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
        else:
            client = bigquery.Client(project=PROJECT_ID)
        
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        LIMIT 500000
        """
        
        df = client.query(query).to_dataframe()
        
        # Procesar fechas
        if 'modif_date' in df.columns:
            df['modif_date'] = pd.to_datetime(df['modif_date'], errors='coerce', utc=True)
            df['modif_date'] = df['modif_date'].dt.tz_convert(None)
        
        # Limpiar datos
        if 'doc_name' in df.columns:
            df['doc_name_clean'] = df['doc_name'].fillna('Sin nombre')
        
        if 'classifications' in df.columns:
            df['classifications_clean'] = df['classifications'].fillna('Sin clasificación')
        
        return df, None
        
    except Exception as e:
        return None, str(e)

def create_classification_chart(data):
    """Crea gráfico de clasificaciones con colores fijos"""
    classification_data = data.get('classification_distribution', {}).get('data', {})
    
    if not classification_data:
        return None
    
    # Crear listas ordenadas
    classifications = list(classification_data.keys())
    values = list(classification_data.values())
    colors = [CLASSIFICATION_COLORS.get(str(c), '#808080') for c in classifications]
    
    # Crear labels descriptivos
    labels = []
    for c in classifications:
        if c == '1.0':
            labels.append('Clasificación 1 (Público)')
        elif c == '2.0':
            labels.append('Clasificación 2 (Confidencial)')
        elif c == '3.0':
            labels.append('Clasificación 3 (Reservado)')
        else:
            labels.append(f'Clasificación {c}')
    
    fig = px.bar(
        x=values,
        y=labels,
        orientation='h',
        title="📋 Distribución de Templates por Clasificación",
        labels={'x': 'Cantidad de Templates', 'y': 'Clasificación'},
        color=classifications,
        color_discrete_map={str(k): v for k, v in CLASSIFICATION_COLORS.items()}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_counter_chart(data):
    """Crea gráfico de contadores"""
    counter_stats = data.get('counter_analysis', {}).get('general_stats', [])[:10]
    
    if not counter_stats:
        return None
    
    df = pd.DataFrame(counter_stats)
    
    fig = px.bar(
        df,
        x='percentage',
        y='clean_name',
        orientation='h',
        title="🔢 Top 10 Contadores - % de Documentos con Datos",
        labels={'percentage': '% de Documentos', 'clean_name': 'Tipo de Contador'},
        color='percentage',
        color_continuous_scale='viridis',
        text='documents_with_data'
    )
    
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(height=500)
    
    return fig

def create_credit_cards_chart(data):
    """Crea gráfico específico de tarjetas de crédito"""
    credit_cards = data.get('counter_analysis', {}).get('specific_categories', {}).get('credit_cards', [])
    
    if not credit_cards:
        return None
    
    df = pd.DataFrame(credit_cards)
    
    fig = px.bar(
        df,
        x='percentage',
        y='clean_name',
        orientation='h',
        title="💳 Tarjetas de Crédito - % de Documentos con Datos",
        labels={'percentage': '% de Documentos', 'clean_name': 'Tipo de Tarjeta'},
        color='total_occurrences',
        color_continuous_scale='blues'
    )
    
    fig.update_layout(height=300)
    return fig

def create_pii_chart(data):
    """Crea gráfico específico de información personal"""
    pii_data = data.get('counter_analysis', {}).get('specific_categories', {}).get('pii_data', [])
    
    if not pii_data:
        return None
    
    df = pd.DataFrame(pii_data)
    
    fig = px.bar(
        df,
        x='percentage',
        y='clean_name',
        orientation='h',
        title="🔒 Información Personal (PII) - % de Documentos con Datos",
        labels={'percentage': '% de Documentos', 'clean_name': 'Tipo de Información'},
        color='total_occurrences',
        color_continuous_scale='reds'
    )
    
    fig.update_layout(height=400)
    return fig

def create_temporal_chart(data):
    """Crea gráfico temporal de los últimos años"""
    temporal_data = data.get('temporal_analysis', {}).get('yearly_data', [])
    
    if not temporal_data:
        return None
    
    # Filtrar años válidos
    df = pd.DataFrame(temporal_data)
    df = df[(df['year'] >= 2015) & (df['year'] <= 2025)]
    
    fig = px.line(
        df,
        x='year',
        y='count',
        title="📈 Evolución Temporal de Templates (Últimos Años)",
        labels={'year': 'Año', 'count': 'Cantidad de Templates'},
        markers=True
    )
    
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(height=400)
    
    return fig

def create_document_names_chart(data):
    """Crea gráfico de nombres de documentos más comunes"""
    doc_analysis = data.get('document_analysis', {})
    top_names = doc_analysis.get('top_names', [])
    
    if not top_names:
        return None
    
    df = pd.DataFrame(top_names[:10])  # Top 10
    
    fig = px.bar(
        df,
        x='frequency',
        y='document_name',
        orientation='h',
        title="📄 Top 10 Nombres de Documentos Más Comunes",
        labels={'frequency': 'Frecuencia', 'document_name': 'Nombre del Documento'},
        color='frequency',
        color_continuous_scale='greens'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_top_documents_by_classification_chart(data, classification):
    """Crea gráfico de top documentos para una clasificación específica"""
    top_docs = data.get('top_documents', {}).get('by_classification', {})
    
    if str(classification) not in top_docs:
        return None
    
    class_data = top_docs[str(classification)]
    top_documents = class_data.get('top_documents', [])[:10]  # Top 10
    
    if not top_documents:
        return None
    
    df = pd.DataFrame(top_documents)
    
    # Usar el color correspondiente a la clasificación
    color = CLASSIFICATION_COLORS.get(str(classification), '#808080')
    
    fig = px.bar(
        df,
        x='frequency',
        y='document_name',
        orientation='h',
        title=f"📋 Top Documentos - Clasificación {classification}",
        labels={'frequency': 'Frecuencia', 'document_name': 'Nombre del Documento'},
        color_discrete_sequence=[color]
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def main():
    # Cargar datos
    data = load_metrics()
    if data is None:
        st.stop()
    
    st.title("📊 Dashboard de Análisis de Templates")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🔧 Configuración del Dashboard")
    st.sidebar.markdown("### Fuente de Datos")
    
    # Selector de modo
    mode = st.sidebar.radio(
        "Selecciona el modo de visualización:",
        ["📊 Vista Rápida (Métricas Pre-calculadas)", "🔍 Exploración en Tiempo Real"]
    )
    
    if mode == "📊 Vista Rápida (Métricas Pre-calculadas)":
        st.sidebar.success("✅ Usando métricas pre-calculadas (Carga Instantánea)")
        timestamp = data.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            st.sidebar.info(f"💡 Datos actualizados: {timestamp[:10]} {timestamp[11:19]}")
        
        # Métricas principales
        st.header("📈 Métricas Principales")
        
        main_metrics = data.get('main_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Templates", f"{main_metrics.get('total_templates', 0):,}")
        
        with col2:
            st.metric("🏷️ Clasificaciones", main_metrics.get('unique_classifications', 0))
        
        with col3:
            st.metric("🔢 Contadores", main_metrics.get('counter_columns_detected', 0))
        
        with col4:
            date_range = main_metrics.get('date_range', {})
            st.metric("📅 Docs con Fecha", f"{date_range.get('valid_dates_count', 0):,}")
        
        st.markdown("---")
        
        # Tabs para las diferentes visualizaciones
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Clasificaciones", 
            "📋 Top Documentos",
            "🔢 Contadores", 
            "📈 Análisis Temporal", 
            "📄 Documentos Generales"
        ])
        
        with tab1:
            st.header("📋 Análisis por Clasificación de Seguridad")
            
            # Gráfico principal
            fig_class = create_classification_chart(data)
            if fig_class:
                st.plotly_chart(fig_class, use_container_width=True)
            
            # Análisis detallado
            st.subheader("🔍 Análisis Detallado por Clasificación")
            
            classification_data = data.get('classification_distribution', {}).get('data', {})
            total_docs = sum(classification_data.values()) if classification_data else 1
            
            col1, col2, col3 = st.columns(3)
            
            # Clasificación 1.0
            if '1.0' in classification_data:
                with col1:
                    st.markdown("### 🟡 Clasificación 1 - Público")
                    count_1 = classification_data['1.0']
                    percentage_1 = (count_1 / total_docs) * 100
                    st.metric("📄 Templates", f"{count_1:,}")
                    st.metric("📊 Porcentaje", f"{percentage_1:.1f}%")
                    st.markdown("**Descripción:** Información de acceso público sin restricciones")
            
            # Clasificación 2.0
            if '2.0' in classification_data:
                with col2:
                    st.markdown("### 🟠 Clasificación 2 - Confidencial")
                    count_2 = classification_data['2.0']
                    percentage_2 = (count_2 / total_docs) * 100
                    st.metric("📄 Templates", f"{count_2:,}")
                    st.metric("📊 Porcentaje", f"{percentage_2:.1f}%")
                    st.markdown("**Descripción:** Información confidencial con acceso restringido")
            
            # Clasificación 3.0
            if '3.0' in classification_data:
                with col3:
                    st.markdown("### 🔴 Clasificación 3 - Reservado")
                    count_3 = classification_data['3.0']
                    percentage_3 = (count_3 / total_docs) * 100
                    st.metric("📄 Templates", f"{count_3:,}")
                    st.metric("📊 Porcentaje", f"{percentage_3:.1f}%")
                    st.markdown("**Descripción:** Información altamente sensible y reservada")
        
        with tab2:
            st.header("📋 Top Documentos por Clasificación")
            
            # Verificar si tenemos datos de top documentos
            top_docs = data.get('top_documents', {})
            by_classification = top_docs.get('by_classification', {})
            
            if by_classification:
                # Selector de clasificación
                available_classifications = list(by_classification.keys())
                selected_classification = st.selectbox(
                    "Selecciona una clasificación:",
                    available_classifications,
                    format_func=lambda x: f"Clasificación {x} ({'Público' if x=='1.0' else 'Confidencial' if x=='2.0' else 'Reservado'})"
                )
                
                if selected_classification:
                    class_data = by_classification[selected_classification]
                    
                    # Métricas de la clasificación
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Total Documentos", f"{class_data.get('classification_total_docs', 0):,}")
                    with col2:
                        st.metric("📊 % del Total", f"{class_data.get('classification_percentage', 0):.1f}%")
                    with col3:
                        top_documents = class_data.get('top_documents', [])
                        if top_documents:
                            total_counters = sum([doc.get('total_counters', 0) for doc in top_documents[:5]])
                            st.metric("🔢 Contadores (Top 5)", f"{total_counters:,}")
                    
                    # Gráfico de top documentos para la clasificación
                    fig_top_docs = create_top_documents_by_classification_chart(data, selected_classification)
                    if fig_top_docs:
                        st.plotly_chart(fig_top_docs, use_container_width=True)
                    
                    # Tabla detallada
                    if top_documents:
                        st.subheader("📋 Detalle de Top Documentos")
                        
                        # Preparar datos para la tabla
                        table_data = []
                        for doc in top_documents[:10]:
                            table_data.append({
                                'Documento': doc.get('document_name', ''),
                                'Frecuencia': f"{doc.get('frequency', 0):,}",
                                '% en Clasificación': f"{doc.get('percentage_in_classification', 0):.2f}%",
                                '% del Total': f"{doc.get('percentage_in_total', 0):.2f}%",
                                'Total Contadores': f"{doc.get('total_counters', 0):,}",
                                'Promedio Contadores': f"{doc.get('average_counters_per_instance', 0):.2f}"
                            })
                        
                        df_table = pd.DataFrame(table_data)
                        st.dataframe(df_table, hide_index=True, use_container_width=True)
            else:
                st.info("📊 Los datos de top documentos por clasificación no están disponibles.")
                
                # Mostrar top documentos generales como alternativa
                general_docs = top_docs.get('general', [])
                if general_docs:
                    st.subheader("🏆 Top Documentos Generales")
                    
                    df_general = pd.DataFrame(general_docs[:10])
                    
                    fig_general = px.bar(
                        df_general,
                        x='frequency',
                        y='document_name',
                        orientation='h',
                        title="Top 10 Documentos Más Frecuentes",
                        color='frequency',
                        color_continuous_scale='viridis'
                    )
                    fig_general.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_general, use_container_width=True)
        
        with tab3:
            st.header("🔢 Análisis de Contadores de Datos Sensibles")
            
            # Gráfico general
            fig_counters = create_counter_chart(data)
            if fig_counters:
                st.plotly_chart(fig_counters, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💳 Tarjetas de Crédito")
                fig_cc = create_credit_cards_chart(data)
                if fig_cc:
                    st.plotly_chart(fig_cc, use_container_width=True)
                
                # Métricas de tarjetas
                credit_cards = data.get('counter_analysis', {}).get('specific_categories', {}).get('credit_cards', [])
                for card in credit_cards:
                    st.metric(
                        f"🔢 {card['clean_name']}",
                        f"{card['total_occurrences']:,}",
                        f"{card['documents_with_data']:,} docs ({card['percentage']:.1f}%)"
                    )
            
            with col2:
                st.subheader("🔒 Información Personal (PII)")
                fig_pii = create_pii_chart(data)
                if fig_pii:
                    st.plotly_chart(fig_pii, use_container_width=True)
                
                # Top 3 PII
                pii_data = data.get('counter_analysis', {}).get('specific_categories', {}).get('pii_data', [])
                top_pii = sorted(pii_data, key=lambda x: x['percentage'], reverse=True)[:3]
                for pii in top_pii:
                    st.metric(
                        f"🔒 {pii['clean_name']}",
                        f"{pii['total_occurrences']:,}",
                        f"{pii['documents_with_data']:,} docs ({pii['percentage']:.1f}%)"
                    )
        
        with tab4:
            st.header("📈 Análisis Temporal")
            
            fig_temporal = create_temporal_chart(data)
            if fig_temporal:
                st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Estadísticas temporales
            temporal_data = data.get('temporal_analysis', {}).get('yearly_data', [])
            if temporal_data:
                col1, col2, col3 = st.columns(3)
                
                # Encontrar año con más templates
                max_year_data = max(temporal_data, key=lambda x: x['count'])
                
                with col1:
                    st.metric("📊 Año con Más Templates", 
                             str(max_year_data['year']), 
                             f"{max_year_data['count']:,} templates")
                
                with col2:
                    # Crecimiento reciente
                    recent_years = [d for d in temporal_data if d['year'] >= 2024]
                    if len(recent_years) >= 2:
                        recent_years.sort(key=lambda x: x['year'])
                        growth = ((recent_years[-1]['count'] - recent_years[-2]['count']) / recent_years[-2]['count']) * 100
                        st.metric("📈 Crecimiento Reciente", 
                                 f"{growth:.1f}%", 
                                 f"+{recent_years[-1]['count'] - recent_years[-2]['count']:,}")
                
                with col3:
                    # Promedio últimos 3 años
                    recent_3_years = [d for d in temporal_data if d['year'] >= 2023]
                    if recent_3_years:
                        avg = sum(d['count'] for d in recent_3_years) / len(recent_3_years)
                        st.metric("📊 Promedio Últimos Años", f"{avg:,.0f}")
        
        with tab5:
            st.header("📄 Análisis de Nombres de Documentos")
            
            fig_names = create_document_names_chart(data)
            if fig_names:
                st.plotly_chart(fig_names, use_container_width=True)
            
            # Estadísticas de documentos
            doc_analysis = data.get('document_analysis', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📄 Total Documentos", f"{doc_analysis.get('total_documents', 0):,}")
            
            with col2:
                st.metric("📝 Nombres Únicos", f"{doc_analysis.get('unique_names', 0):,}")
            
            with col3:
                # Documento más común
                top_names = doc_analysis.get('top_names', [])
                if top_names:
                    top_doc = top_names[0]
                    doc_name = top_doc["document_name"]
                    display_name = doc_name[:20] + "..." if len(doc_name) > 20 else doc_name
                    st.metric("🏆 Más Común", display_name, f"{top_doc['frequency']:,}")
    
    else:  # Modo exploración en tiempo real
        st.sidebar.warning("⚠️ Modo en tiempo real - Requiere conexión a BigQuery")
        
        # Verificar credenciales
        credentials, cred_error = setup_bigquery_credentials()
        
        if cred_error:
            st.sidebar.error(f"❌ {cred_error}")
            st.error("🔧 **Configuración de Credenciales Requerida**")
            st.markdown("""
            Para usar el modo de exploración en tiempo real, necesitas configurar las credenciales de BigQuery:
            
            **Opciones disponibles:**
            1. **Streamlit Cloud:** Configura `st.secrets` 
            2. **Local:** Variables de entorno o archivo JSON
            3. **Otras plataformas:** Variables de entorno
            
            Mientras tanto, puedes usar la **Vista Rápida** con métricas pre-calculadas.
            """)
            st.stop()
        else:
            st.sidebar.success("✅ Credenciales configuradas")
        
        # Botón para cargar muestra
        if st.sidebar.button("🔄 Cargar Muestra de Datos (500,000 registros)", type="primary"):
            with st.spinner("🔄 Cargando muestra desde BigQuery..."):
                sample_df, error = load_sample_data_from_bigquery()
                
                if error:
                    st.error(f"Error cargando datos: {error}")
                else:
                    st.success(f"✅ Muestra cargada: {len(sample_df):,} registros")
                    st.session_state['sample_df'] = sample_df
        
        # Mostrar datos de muestra si están disponibles
        if 'sample_df' in st.session_state:
            sample_df = st.session_state['sample_df']
            
            st.header("🔍 Exploración de Datos en Tiempo Real")
            main_metrics = data.get('main_metrics', {})
            total_templates = main_metrics.get('total_templates', 0)
            st.info(f"📊 **Muestra actual:** {len(sample_df):,} registros de {total_templates:,} totales")
            
            # Filtros interactivos
            col1, col2 = st.columns(2)
            
            with col1:
                if 'classifications_clean' in sample_df.columns:
                    classifications = sample_df['classifications_clean'].unique()
                    selected_classifications = st.multiselect(
                        "🏷️ Filtrar por Clasificación:",
                        options=classifications,
                        default=classifications
                    )
                else:
                    selected_classifications = []
            
            with col2:
                if 'modif_date' in sample_df.columns and sample_df['modif_date'].notna().any():
                    date_range = st.date_input(
                        "📅 Filtrar por Fecha:",
                        value=(sample_df['modif_date'].min().date(), sample_df['modif_date'].max().date()),
                        min_value=sample_df['modif_date'].min().date(),
                        max_value=sample_df['modif_date'].max().date()
                    )
            
            # Aplicar filtros
            filtered_df = sample_df.copy()
            
            if selected_classifications and 'classifications_clean' in sample_df.columns:
                filtered_df = filtered_df[filtered_df['classifications_clean'].isin(selected_classifications)]
            
            st.subheader(f"📋 Datos Filtrados ({len(filtered_df):,} registros)")
            
            # Mostrar información de columnas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Información del Dataset:**")
                st.write(f"- **Filas:** {len(filtered_df):,}")
                st.write(f"- **Columnas:** {len(filtered_df.columns)}")
                st.write(f"- **Memoria:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            with col2:
                st.markdown("**🔍 Estadísticas Rápidas:**")
                if 'classifications_clean' in filtered_df.columns:
                    class_counts = filtered_df['classifications_clean'].value_counts()
                    for classification, count in class_counts.items():
                        color = CLASSIFICATION_COLORS.get(str(classification), '#808080')
                        st.markdown(f"- **Clasificación {classification}:** {count:,} ({count/len(filtered_df)*100:.1f}%)")
            
            # Tabla de datos
            st.dataframe(
                filtered_df.head(500),  # Mostrar hasta 500 registros
                use_container_width=True,
                height=400
            )
            
            # Opción de descarga
            if st.button("📥 Descargar datos filtrados como CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="💾 Descargar CSV",
                    data=csv,
                    file_name=f"templates_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("👆 Haz clic en 'Cargar Muestra de Datos' para explorar los datos en tiempo real")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("*📊 Dashboard optimizado con métricas pre-calculadas*")
    
    with col2:
        timestamp = data.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            st.markdown(f"*⚡ Última actualización: {timestamp[:10]}*")
    
    with col3:
        st.markdown("*🚀 Creado con Streamlit y BigQuery*")

if __name__ == "__main__":
    main()