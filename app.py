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

# Métricas pre-calculadas (basadas en tu archivo JSON)
PRECALCULATED_METRICS = {
    "timestamp": "2025-06-13T16:54:41.051340",
    "total_templates": 1640532,
    "unique_classifications": 3,
    "counter_columns_detected": 11,
    "date_range": {
        "min_date": "1979-11-30",
        "max_date": "2098-01-01",
        "valid_dates_count": 1640532
    },
    "classification_distribution": {
        "3.0": 790017,
        "1.0": 568832,
        "2.0": 281683
    },
    "detailed_classifications": {
        "1.0": {"count": 568832, "percentage": 34.7, "total_counters": 0, "unique_docs": 0},
        "2.0": {"count": 281683, "percentage": 17.2, "total_counters": 0, "unique_docs": 0},
        "3.0": {"count": 790017, "percentage": 48.1, "total_counters": 0, "unique_docs": 0}
    },
    "counter_stats": [
        {"counter_name": "analysis_total_counters", "clean_name": "Analysis Total", "documents_with_data": 1561140, "percentage": 95.16, "total_sum": 1682645244, "max_value": 6468583, "average": 1025.67},
        {"counter_name": "pii_personal_name_count", "clean_name": "PII Personal Name", "documents_with_data": 1539598, "percentage": 93.85, "total_sum": 1369715115, "max_value": 4508963, "average": 841.04},
        {"counter_name": "pii_ced_ecu_count", "clean_name": "PII Cédula Ecuador", "documents_with_data": 672786, "percentage": 41.01, "total_sum": 241174770, "max_value": 1959511, "average": 148.09},
        {"counter_name": "pii_email_count", "clean_name": "PII Email", "documents_with_data": 408361, "percentage": 24.89, "total_sum": 37132328, "max_value": 223906, "average": 22.8},
        {"counter_name": "pii_ruc_ecu_count", "clean_name": "PII RUC Ecuador", "documents_with_data": 151994, "percentage": 9.26, "total_sum": 7093122, "max_value": 86666, "average": 4.36},
        {"counter_name": "cc_visa_count", "clean_name": "Tarjetas Visa", "documents_with_data": 41912, "percentage": 2.55, "total_sum": 10073870, "max_value": 728012, "average": 6.19},
        {"counter_name": "cc_mastercard_count", "clean_name": "Tarjetas Mastercard", "documents_with_data": 26260, "percentage": 1.6, "total_sum": 4110778, "max_value": 260879, "average": 2.52},
        {"counter_name": "pii_phone_number_count", "clean_name": "PII Teléfono", "documents_with_data": 12691, "percentage": 0.77, "total_sum": 141629, "max_value": 8495, "average": 0.09},
        {"counter_name": "pii_address_count", "clean_name": "PII Dirección", "documents_with_data": 4738, "percentage": 0.29, "total_sum": 13348, "max_value": 481, "average": 0.01},
        {"counter_name": "cc_discover_count", "clean_name": "Tarjetas Discover", "documents_with_data": 2893, "percentage": 0.18, "total_sum": 174702, "max_value": 8463, "average": 0.11},
        {"counter_name": "cc_diners_club_count", "clean_name": "Tarjetas Diners", "documents_with_data": 1899, "percentage": 0.12, "total_sum": 66826, "max_value": 11626, "average": 0.04}
    ],
    "credit_cards": [
        {"counter_name": "cc_discover_count", "clean_name": "Discover", "total_occurrences": 174702, "documents_with_data": 2893, "percentage": 0.18},
        {"counter_name": "cc_visa_count", "clean_name": "Visa", "total_occurrences": 10073870, "documents_with_data": 41912, "percentage": 2.55},
        {"counter_name": "cc_diners_club_count", "clean_name": "Diners Club", "total_occurrences": 66826, "documents_with_data": 1899, "percentage": 0.12},
        {"counter_name": "cc_mastercard_count", "clean_name": "Mastercard", "total_occurrences": 4110778, "documents_with_data": 26260, "percentage": 1.6}
    ],
    "pii_data": [
        {"counter_name": "pii_address_count", "clean_name": "Direcciones", "total_occurrences": 13348, "documents_with_data": 4738, "percentage": 0.29},
        {"counter_name": "pii_ruc_ecu_count", "clean_name": "RUC Ecuador", "total_occurrences": 7093122, "documents_with_data": 151994, "percentage": 9.26},
        {"counter_name": "pii_ced_ecu_count", "clean_name": "Cédulas Ecuador", "total_occurrences": 241174770, "documents_with_data": 672786, "percentage": 41.01},
        {"counter_name": "pii_phone_number_count", "clean_name": "Teléfonos", "total_occurrences": 141629, "documents_with_data": 12691, "percentage": 0.77},
        {"counter_name": "pii_personal_name_count", "clean_name": "Nombres Personales", "total_occurrences": 1369715115, "documents_with_data": 1539598, "percentage": 93.85},
        {"counter_name": "pii_email_count", "clean_name": "Emails", "total_occurrences": 37132328, "documents_with_data": 408361, "percentage": 24.89}
    ],
    "temporal_data": {
        "yearly_totals": {
            "2019": 66636, "2020": 92258, "2021": 128161, "2022": 150664, 
            "2023": 192702, "2024": 347215, "2025": 407830
        }
    },
    "top_document_names": [
        {"document_name": "UNGENERALIZABLE", "frequency": 176136},
        {"document_name": "rpa signed", "frequency": 46770},
        {"document_name": "bitacora availability ecu", "frequency": 25123},
        {"document_name": "tarjeta registro firma", "frequency": 23898},
        {"document_name": "prc comprobante control", "frequency": 17566},
        {"document_name": "xxxxxxxx", "frequency": 14613},
        {"document_name": "oficio nro urr signed", "frequency": 14536},
        {"document_name": "notice", "frequency": 12316},
        {"document_name": "ruc", "frequency": 11163},
        {"document_name": "solicitud servicios electronicos", "frequency": 10394}
    ]
}

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
        LIMIT 1000
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

def create_classification_chart():
    """Crea gráfico de clasificaciones con colores fijos"""
    data = PRECALCULATED_METRICS["classification_distribution"]
    
    # Crear listas ordenadas
    classifications = list(data.keys())
    values = list(data.values())
    colors = [CLASSIFICATION_COLORS[str(c)] for c in classifications]
    
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

def create_counter_chart():
    """Crea gráfico de contadores"""
    data = PRECALCULATED_METRICS["counter_stats"][:10]  # Top 10
    
    df = pd.DataFrame(data)
    
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

def create_credit_cards_chart():
    """Crea gráfico específico de tarjetas de crédito"""
    data = PRECALCULATED_METRICS["credit_cards"]
    df = pd.DataFrame(data)
    
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

def create_pii_chart():
    """Crea gráfico específico de información personal"""
    data = PRECALCULATED_METRICS["pii_data"]
    df = pd.DataFrame(data)
    
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

def create_temporal_chart():
    """Crea gráfico temporal de los últimos años"""
    data = PRECALCULATED_METRICS["temporal_data"]["yearly_totals"]
    
    years = list(data.keys())
    values = list(data.values())
    
    fig = px.line(
        x=years,
        y=values,
        title="📈 Evolución Temporal de Templates (Últimos Años)",
        labels={'x': 'Año', 'y': 'Cantidad de Templates'},
        markers=True
    )
    
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(height=400)
    
    return fig

def create_document_names_chart():
    """Crea gráfico de nombres de documentos más comunes"""
    data = PRECALCULATED_METRICS["top_document_names"]
    df = pd.DataFrame(data)
    
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

def main():
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
        st.sidebar.info("💡 Los datos fueron calculados el 13 de Junio de 2025 a las 16:54")
        
        # Métricas principales
        st.header("📈 Métricas Principales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Templates", f"{PRECALCULATED_METRICS['total_templates']:,}")
        
        with col2:
            st.metric("🏷️ Clasificaciones", PRECALCULATED_METRICS['unique_classifications'])
        
        with col3:
            st.metric("🔢 Contadores", PRECALCULATED_METRICS['counter_columns_detected'])
        
        with col4:
            st.metric("📅 Período", f"{PRECALCULATED_METRICS['date_range']['valid_dates_count']:,} fechas")
        
        st.markdown("---")
        
        # Tabs para las diferentes visualizaciones
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Clasificaciones", 
            "🔢 Contadores", 
            "📈 Análisis Temporal", 
            "📄 Documentos"
        ])
        
        with tab1:
            st.header("📋 Análisis por Clasificación de Seguridad")
            
            # Gráfico principal
            fig_class = create_classification_chart()
            st.plotly_chart(fig_class, use_container_width=True)
            
            # Análisis detallado
            st.subheader("🔍 Análisis Detallado por Clasificación")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🟡 Clasificación 1 - Público")
                st.metric("📄 Templates", f"{PRECALCULATED_METRICS['detailed_classifications']['1.0']['count']:,}")
                st.metric("📊 Porcentaje", f"{PRECALCULATED_METRICS['detailed_classifications']['1.0']['percentage']:.1f}%")
                st.markdown("**Descripción:** Información de acceso público sin restricciones")
            
            with col2:
                st.markdown("### 🟠 Clasificación 2 - Confidencial")
                st.metric("📄 Templates", f"{PRECALCULATED_METRICS['detailed_classifications']['2.0']['count']:,}")
                st.metric("📊 Porcentaje", f"{PRECALCULATED_METRICS['detailed_classifications']['2.0']['percentage']:.1f}%")
                st.markdown("**Descripción:** Información confidencial con acceso restringido")
            
            with col3:
                st.markdown("### 🔴 Clasificación 3 - Reservado")
                st.metric("📄 Templates", f"{PRECALCULATED_METRICS['detailed_classifications']['3.0']['count']:,}")
                st.metric("📊 Porcentaje", f"{PRECALCULATED_METRICS['detailed_classifications']['3.0']['percentage']:.1f}%")
                st.markdown("**Descripción:** Información altamente sensible y reservada")
        
        with tab2:
            st.header("🔢 Análisis de Contadores de Datos Sensibles")
            
            # Gráfico general
            fig_counters = create_counter_chart()
            st.plotly_chart(fig_counters, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💳 Tarjetas de Crédito")
                fig_cc = create_credit_cards_chart()
                st.plotly_chart(fig_cc, use_container_width=True)
                
                # Métricas de tarjetas
                for card in PRECALCULATED_METRICS["credit_cards"]:
                    st.metric(
                        f"🔢 {card['clean_name']}",
                        f"{card['total_occurrences']:,}",
                        f"{card['documents_with_data']:,} docs ({card['percentage']:.1f}%)"
                    )
            
            with col2:
                st.subheader("🔒 Información Personal (PII)")
                fig_pii = create_pii_chart()
                st.plotly_chart(fig_pii, use_container_width=True)
                
                # Top 3 PII
                top_pii = sorted(PRECALCULATED_METRICS["pii_data"], key=lambda x: x['percentage'], reverse=True)[:3]
                for pii in top_pii:
                    st.metric(
                        f"🔒 {pii['clean_name']}",
                        f"{pii['total_occurrences']:,}",
                        f"{pii['documents_with_data']:,} docs ({pii['percentage']:.1f}%)"
                    )
        
        with tab3:
            st.header("📈 Análisis Temporal")
            
            fig_temporal = create_temporal_chart()
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Estadísticas temporales
            col1, col2, col3 = st.columns(3)
            
            yearly_data = PRECALCULATED_METRICS["temporal_data"]["yearly_totals"]
            years = list(yearly_data.keys())
            
            with col1:
                st.metric("📊 Año con Más Templates", "2025", f"{yearly_data['2025']:,} templates")
            
            with col2:
                # Crecimiento entre 2024 y 2025
                growth = ((yearly_data['2025'] - yearly_data['2024']) / yearly_data['2024']) * 100
                st.metric("📈 Crecimiento 2024-2025", f"{growth:.1f}%", f"+{yearly_data['2025'] - yearly_data['2024']:,}")
            
            with col3:
                # Promedio últimos 3 años
                recent_avg = (yearly_data['2023'] + yearly_data['2024'] + yearly_data['2025']) / 3
                st.metric("📊 Promedio Últimos 3 Años", f"{recent_avg:,.0f}")
        
        with tab4:
            st.header("📄 Análisis de Nombres de Documentos")
            
            fig_names = create_document_names_chart()
            st.plotly_chart(fig_names, use_container_width=True)
            
            # Estadísticas de documentos
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📄 Total Documentos", f"{PRECALCULATED_METRICS['total_templates']:,}")
            
            with col2:
                st.metric("📝 Nombres Únicos", "425,544")
            
            with col3:
                # Documento más común
                top_doc = PRECALCULATED_METRICS["top_document_names"][0]
                st.metric("🏆 Más Común", top_doc["document_name"][:20] + "...", f"{top_doc['frequency']:,}")
    
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
        if st.sidebar.button("🔄 Cargar Muestra de Datos (1,000 registros)", type="primary"):
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
            st.info(f"📊 **Muestra actual:** {len(sample_df):,} registros de {PRECALCULATED_METRICS['total_templates']:,} totales")
            
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
        st.markdown(f"*⚡ Última actualización: {PRECALCULATED_METRICS['timestamp'][:10]}*")
    
    with col3:
        st.markdown("*🚀 Creado con Streamlit y BigQuery*")

if __name__ == "__main__":
    main()