import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Dashboard ML Accuracy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard de Accuracy - Modelos ML")

# Función para cargar datos
@st.cache_data
def load_data():
    import os
    
    # Mostrar estructura de archivos para debug
    st.sidebar.markdown("### 🔍 Debug Info")
    current_dir = os.getcwd()
    st.sidebar.text(f"Directorio actual: {current_dir}")
    
    # Listar archivos en directorio actual
    files_in_current = os.listdir('.')
    st.sidebar.text("Archivos en directorio actual:")
    for f in files_in_current:
        st.sidebar.text(f"  - {f}")
    
    # Verificar si existe carpeta data
    if os.path.exists('data'):
        st.sidebar.text("Archivos en carpeta data/:")
        files_in_data = os.listdir('data')
        for f in files_in_data:
            st.sidebar.text(f"  - {f}")
    else:
        st.sidebar.error("❌ Carpeta 'data/' no existe")
    
    try:
        # Intentar diferentes rutas solo para ml_sensitivity
        paths_to_try = [
            'data/ml_sensitivity.parquet',
            'ml_sensitivity.parquet',
            './data/ml_sensitivity.parquet'
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                st.sidebar.success(f"✅ Archivo encontrado en: {path}")
                sensitivity_df = pd.read_parquet(path)
                return sensitivity_df
        
        # Si no encuentra el archivo, mostrar error detallado
        st.error("❌ No se encontró el archivo ml_sensitivity.parquet")
        st.info("""
        **Soluciones posibles:**
        1. Asegúrate de que el archivo esté nombrado exactamente como:
           - `ml_sensitivity.parquet`
        
        2. Verifica la estructura de carpetas:
           ```
           streamlit_app/
           ├── app.py
           └── data/
               └── ml_sensitivity.parquet
           ```
        
        3. Si el archivo está en el directorio raíz, muévelo a la carpeta 'data/'
        """)
        
        return None
        
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

# Cargar datos
sensitivity_data = load_data()

# Sidebar para navegación
st.sidebar.title("Navegación")
section = st.sidebar.selectbox(
    "Selecciona sección:",
    ["📋 Resumen General", "🤖 Accuracy por Agente", "🔒 Accuracy por Confidentiality", 
     "📈 Comparación de Versiones", "🧹 Análisis de Vectores", "📊 Explorar Datos", "📝 Análisis Template"]
)

# Función para crear matriz de confusión heatmap
def create_confusion_heatmap(matrix, title, labels=None):
    if labels is None:
        labels = [f"Clase {i}" for i in range(len(matrix))]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=np.round(matrix, 2),
        texttemplate="%{text}%",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Real",
        height=400
    )
    
    return fig

# Función para crear gráfico de barras de accuracy
def create_accuracy_bar(accuracies, labels, title):
    fig = go.Figure(data=[
        go.Bar(x=labels, y=accuracies, 
               text=[f"{acc:.3f}" for acc in accuracies],
               textposition='auto',
               marker_color='lightblue')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Categoría",
        yaxis_title="Accuracy",
        height=400
    )
    
    return fig

# SECCIÓN: RESUMEN GENERAL
if section == "📋 Resumen General":
    st.header("Resumen General del Sistema")
    
    # Nota importante sobre la base de cálculo
    st.info("📊 **Nota Importante:** Todos los accuracy mostrados en este dashboard fueron calculados sobre una base de **2,388,716 documentos** que consideran únicamente aquellos con **feedback consistente**.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total de Registros",
            value="2,388,716",
            help="Base total con feedback consistente"
        )
    
    with col2:
        st.metric(
            label="Accuracy Promedio General",
            value="28.18%",
            help="Accuracy con feedback consistente"
        )
    
    with col3:
        st.metric(
            label="Vectores Vacíos",
            value="190,133",
            delta="-7.96%",
            delta_color="inverse",
            help="Vectores que fueron excluidos"
        )
    
    with col4:
        st.metric(
            label="Datos Limpios",
            value="2,198,583",
            help="Datos después de limpiar vectores vacíos"
        )
    
    st.markdown("---")
    
    # Distribución por agente
    st.subheader("Distribución por Agente")
    agent_data = {
        'Agente': ['Windows Agent', 'OneDrive Agent'],
        'Registros': [2374449, 14267],
        'Porcentaje': [99.40, 0.60]
    }
    
    fig_dist = px.pie(
        values=agent_data['Registros'],
        names=agent_data['Agente'],
        title="Distribución de Registros por Agente"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# SECCIÓN: ACCURACY POR AGENTE
elif section == "🤖 Accuracy por Agente":
    st.header("Análisis de Accuracy por Agente")
    
    # Métricas principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Windows Agent Accuracy",
            value="33.12%",
            help="Promedio de accuracy para Windows Agent"
        )
    
    with col2:
        st.metric(
            label="OneDrive Agent Accuracy",
            value="42.57%",
            delta="+9.45%",
            delta_color="normal",
            help="Promedio de accuracy para OneDrive Agent"
        )
    
    # Gráfico comparativo
    accuracies = [0.3312, 0.4257]
    agents = ['Windows Agent', 'OneDrive Agent']
    
    fig_agents = create_accuracy_bar(accuracies, agents, "Comparación de Accuracy por Agente")
    st.plotly_chart(fig_agents, use_container_width=True)
    
    # Matrices de confusión
    st.subheader("Matrices de Confusión por Agente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Windows Agent**")
        windows_matrix = np.array([
            [38.89, 28.75, 22.21, 10.14],
            [16.01, 46.29, 23.73, 13.97],
            [31.60, 28.97, 24.13, 15.30],
            [8.33, 51.60, 16.88, 23.18]
        ])
        fig_windows = create_confusion_heatmap(windows_matrix, "Windows Agent - Confusion Matrix")
        st.plotly_chart(fig_windows, use_container_width=True)
    
    with col2:
        st.markdown("**OneDrive Agent**")
        onedrive_matrix = np.array([
            [6.54, 14.95, 76.64, 1.87],
            [12.72, 58.43, 15.58, 13.26],
            [5.91, 83.75, 5.29, 5.04],
            [0.00, 0.00, 0.00, 100.00]
        ])
        fig_onedrive = create_confusion_heatmap(onedrive_matrix, "OneDrive Agent - Confusion Matrix")
        st.plotly_chart(fig_onedrive, use_container_width=True)

# SECCIÓN: ACCURACY POR CONFIDENTIALITY
elif section == "🔒 Accuracy por Confidentiality":
    st.header("Análisis por Confidentiality Model")
    
    # Métricas de confidentiality
    conf_data = {
        'Modelo': ['Template', 'Sensitivity', 'Risk', 'Regex'],
        'Registros': [164418, 51451, 74996, 48214],
        'Accuracy': [0.2633, np.nan, 0.3247, 0.5172]
    }
    
    # Calcular accuracy real para sensitivity si tenemos los datos
    if sensitivity_data is not None and 'REAL' in sensitivity_data.columns and 'classifications' in sensitivity_data.columns:
        try:
            from sklearn.metrics import confusion_matrix
            
            # Filtrar solo datos de sensitivity
            if 'confidentiality_model_obtained' in sensitivity_data.columns:
                sens_data = sensitivity_data[sensitivity_data['confidentiality_model_obtained'] == 'sensitivity']
                if len(sens_data) > 0:
                    # Usar tu método exacto para calcular accuracy
                    confusion = confusion_matrix(sens_data['REAL'], sens_data['classifications'])
                    accuracy_per_class = np.diag(confusion) / np.sum(confusion, axis=1)
                    # Filtrar NaN y calcular promedio solo de valores válidos
                    valid_accuracies = accuracy_per_class[~np.isnan(accuracy_per_class)]
                    real_sens_accuracy = np.mean(valid_accuracies) if len(valid_accuracies) > 0 else np.nan
                    
                    if not np.isnan(real_sens_accuracy):
                        conf_data['Accuracy'][1] = real_sens_accuracy  # Reemplazar NaN con el accuracy real
                        st.info(f"🔄 Accuracy de Sensitivity calculado del dataset: {real_sens_accuracy:.4f} (promedio de clases válidas)")
                    else:
                        st.warning("⚠️ No se pudo calcular accuracy válido para Sensitivity")
        except Exception as e:
            st.warning(f"No se pudo calcular accuracy real de sensitivity: {e}")
    
    # Métricas por modelo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Template", "26.33%", help="164,418 registros")
    
    with col2:
        if np.isnan(conf_data['Accuracy'][1]):
            st.metric("Sensitivity", "N/A", help="51,451 registros - Contiene NaN")
        else:
            sens_acc_pct = conf_data['Accuracy'][1] * 100
            st.metric("Sensitivity", f"{sens_acc_pct:.2f}%", help=f"51,451 registros - Calculado del dataset")
    
    with col3:
        st.metric("Risk", "32.47%", help="74,996 registros")
    
    with col4:
        st.metric("Regex", "51.72%", help="48,214 registros")
    
    # Gráfico de accuracy por modelo (incluyendo sensitivity si está disponible)
    valid_models = []
    valid_accuracies = []
    
    for i, model in enumerate(conf_data['Modelo']):
        if not np.isnan(conf_data['Accuracy'][i]):
            valid_models.append(model)
            valid_accuracies.append(conf_data['Accuracy'][i])
    
    fig_conf = create_accuracy_bar(valid_accuracies, valid_models, "Accuracy por Confidentiality Model")
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Selector de modelo para ver matriz de confusión
    st.subheader("Matriz de Confusión por Modelo")
    
    model_choice = st.selectbox("Selecciona modelo:", valid_models)
    
    if model_choice == "Template":
        template_matrix = np.array([
            [0.00, 36.84, 18.95, 44.21],
            [0.00, 33.99, 33.50, 32.51],
            [0.00, 28.49, 9.42, 62.08],
            [0.00, 30.86, 7.23, 61.91]
        ])
        fig_template = create_confusion_heatmap(template_matrix, "Template - Confusion Matrix")
        st.plotly_chart(fig_template, use_container_width=True)
    
    elif model_choice == "Sensitivity":
        # Matriz de confusión con NaN reemplazados por 0
        sensitivity_matrix = np.array([
            [0.00, 0.00, 0.00, 0.00],  # NaN → 0
            [21.31, 70.71, 1.94, 6.04],
            [40.31, 41.51, 8.42, 9.76],
            [17.29, 67.29, 0.00, 15.41]
        ])
        fig_sensitivity = create_confusion_heatmap(sensitivity_matrix, "Sensitivity - Confusion Matrix (NaN→0)")
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        st.info("""
        ⚠️ **Nota sobre Sensitivity:** La primera clase (Clase 0) contenía valores NaN 
        que han sido reemplazados por 0 para la visualización. El accuracy se calculó 
        excluyendo los valores NaN: promedio de clases válidas = 31.43%
        """)
    
    elif model_choice == "Risk":
        risk_matrix = np.array([
            [0.11, 5.95, 93.94],
            [0.01, 3.07, 96.92],
            [0.04, 5.73, 94.24]
        ])
        fig_risk = create_confusion_heatmap(risk_matrix, "Risk - Confusion Matrix")
        st.plotly_chart(fig_risk, use_container_width=True)
    
    elif model_choice == "Regex":
        regex_matrix = np.array([
            [66.46, 33.54, 0.00, 0.00],
            [0.00, 75.00, 7.43, 17.56],
            [0.00, 59.55, 38.68, 1.77],
            [0.00, 36.07, 37.18, 26.75]
        ])
        fig_regex = create_confusion_heatmap(regex_matrix, "Regex - Confusion Matrix")
        st.plotly_chart(fig_regex, use_container_width=True)
    
    # Agregar análisis especial para Sensitivity
    st.subheader("⚠️ Análisis Especial: Sensitivity Model")
    
    st.markdown("""
    **Nota importante:** El modelo Sensitivity presenta valores NaN en la primera clase, 
    lo que afecta el cálculo de accuracy promedio. A continuación se muestra el análisis 
    excluyendo los valores NaN.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Registros Sensitivity", "51,451")
    
    with col2:
        # Accuracy excluyendo NaN: promedio de [0.70706988, 0.08418708, 0.15413534]
        accuracy_without_nan = np.mean([0.70706988, 0.08418708, 0.15413534])
        st.metric("Accuracy (sin NaN)", f"{accuracy_without_nan:.4f}")
    
    with col3:
        st.metric("Clases Válidas", "3 de 4")
    
    # Matriz de confusión para Sensitivity (reemplazando NaN con 0)
    sensitivity_matrix = np.array([
        [0.00, 0.00, 0.00, 0.00],  # Primera fila: NaN → 0
        [21.31, 70.71, 1.94, 6.04],
        [40.31, 41.51, 8.42, 9.76],
        [17.29, 67.29, 0.00, 15.41]
    ])
    
    fig_sensitivity = create_confusion_heatmap(sensitivity_matrix, "Sensitivity - Confusion Matrix (NaN → 0)")
    st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    # Accuracy por clase para Sensitivity
    st.markdown("#### Accuracy por Clase - Sensitivity Model")
    
    sensitivity_accuracies = [0.0, 0.7071, 0.0842, 0.1541]  # NaN → 0
    sensitivity_labels = ['Clase 0 (NaN→0)', 'Clase 1', 'Clase 2', 'Clase 3']
    
    fig_sens_acc = create_accuracy_bar(sensitivity_accuracies, sensitivity_labels, "Sensitivity - Accuracy por Clase")
    st.plotly_chart(fig_sens_acc, use_container_width=True)
    
    # Tabla detallada
    sens_detail_df = pd.DataFrame({
        'Clase': ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3'],
        'Accuracy Original': ['NaN', '0.7071', '0.0842', '0.1541'],
        'Accuracy Ajustado': ['0.0000', '0.7071', '0.0842', '0.1541'],
        'Estado': ['Problemática (NaN)', 'Válida', 'Válida', 'Válida']
    })
    
    st.dataframe(sens_detail_df, use_container_width=True)

# SECCIÓN: COMPARACIÓN DE VERSIONES
elif section == "📈 Comparación de Versiones":
    st.header("Comparación entre Versiones de ML")
    
    # Métricas de versiones
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="ML Nueva (v150 + v171)",
            value="29.22%",
            help="339,079 registros total"
        )
        st.caption("v150: 284,354 | v171: 54,725")
    
    with col2:
        st.metric(
            label="ML Anterior (v38 + v72)",
            value="33.52%",
            delta="+4.30%",
            delta_color="normal",
            help="2,049,637 registros total"
        )
        st.caption("v38: 1,060,365 | v72: 989,272")
    
    with col3:
        st.metric(
            label="Datos Limpios (sin vectores vacíos)",
            value="32.64%",
            help="2,198,583 registros"
        )
    
    # Gráfico comparativo de versiones
    versions = ['ML Nueva', 'ML Anterior', 'Datos Limpios']
    version_accuracies = [0.2922, 0.3352, 0.3264]
    
    fig_versions = create_accuracy_bar(version_accuracies, versions, "Comparación de Accuracy por Versión")
    st.plotly_chart(fig_versions, use_container_width=True)
    
    # Accuracy por clase para cada versión
    st.subheader("Accuracy por Clase")
    
    version_choice = st.selectbox("Selecciona versión:", versions)
    
    if version_choice == "ML Nueva":
        new_accuracies = [0.1144, 0.3751, 0.1567, 0.5227]
        class_labels = ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3']
        fig_new = create_accuracy_bar(new_accuracies, class_labels, "ML Nueva - Accuracy por Clase")
        st.plotly_chart(fig_new, use_container_width=True)
    
    elif version_choice == "ML Anterior":
        old_accuracies = [0.4249, 0.4756, 0.2398, 0.2005]
        class_labels = ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3']
        fig_old = create_accuracy_bar(old_accuracies, class_labels, "ML Anterior - Accuracy por Clase")
        st.plotly_chart(fig_old, use_container_width=True)
    
    else:  # Datos Limpios
        clean_accuracies = [0.3984, 0.4273, 0.2387, 0.2411]
        class_labels = ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3']
        fig_clean = create_accuracy_bar(clean_accuracies, class_labels, "Datos Limpios - Accuracy por Clase")
        st.plotly_chart(fig_clean, use_container_width=True)

# SECCIÓN: ANÁLISIS DE VECTORES
elif section == "🧹 Análisis de Vectores":
    st.header("Análisis de Vectores Vacíos")
    
    # Métricas de limpieza
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Original",
            value="2,388,716",
            help="Registros antes de limpiar"
        )
    
    with col2:
        st.metric(
            label="Vectores Vacíos",
            value="190,133",
            delta="-7.96%",
            delta_color="inverse",
            help="Vectores excluidos"
        )
    
    with col3:
        st.metric(
            label="Datos Finales",
            value="2,198,583",
            delta="92.04%",
            delta_color="normal",
            help="Datos después de limpieza"
        )
    
    # Visualización del impacto de la limpieza
    cleaning_data = {
        'Categoría': ['Datos Originales', 'Vectores Vacíos', 'Datos Limpios'],
        'Cantidad': [2388716, 190133, 2198583],
        'Color': ['blue', 'red', 'green']
    }
    
    fig_cleaning = px.bar(
        x=cleaning_data['Categoría'],
        y=cleaning_data['Cantidad'],
        title="Impacto de la Limpieza de Vectores",
        color=cleaning_data['Color'],
        color_discrete_map={'blue': 'lightblue', 'red': 'lightcoral', 'green': 'lightgreen'}
    )
    
    st.plotly_chart(fig_cleaning, use_container_width=True)
    
    # Comparación de accuracy antes y después de limpiar
    st.subheader("Impacto en Accuracy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Accuracy con Vectores Vacíos",
            value="28.18%",
            help="Accuracy con todos los datos"
        )
    
    with col2:
        st.metric(
            label="Accuracy sin Vectores Vacíos",
            value="32.64%",
            delta="+4.46%",
            delta_color="normal",
            help="Accuracy después de limpiar"
        )
    
    # Matriz de confusión de datos limpios
    st.subheader("Matriz de Confusión - Datos Limpios")
    
    clean_matrix = np.array([
        [39.84, 42.73, 23.87, 24.11],
        [17.59, 56.53, 19.12, 6.75],
        [17.06, 51.87, 15.56, 15.51],
        [19.28, 29.45, 20.12, 31.15]
    ])
    
    # Nota: Esta matriz parece tener un formato diferente, ajustando para visualización
    general_matrix = np.array([
        [17.59, 56.53, 19.12, 6.75],
        [17.06, 51.87, 15.56, 15.51],
        [19.28, 29.45, 20.12, 31.15],
        [14.16, 49.84, 12.84, 23.15]
    ])
    
    fig_general = create_confusion_heatmap(general_matrix, "Matriz de Confusión General")
    st.plotly_chart(fig_general, use_container_width=True)

# SECCIÓN: ANÁLISIS TEMPLATE
elif section == "📝 Análisis Template":
    st.header("Análisis de Documentos Template")
    
    st.info("📊 Análisis basado en datos de la base **Relabeling**")
    st.success("🗃️ Base de datos: **Relabeling** - Total de registros: **1,353,568**")
    
    # Análisis TEMPLATE = TRUE (datos fijos)
    st.subheader("🔍 Análisis para TEMPLATE = TRUE")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documentos Template", "107,263")
    
    with col2:
        pct_template = (107263 / 1353568) * 100
        st.metric("% del Total (Relabeling)", f"{pct_template:.2f}%")
    
    with col3:
        st.metric("Nivel de Sensibilidad", "Template")
    
    # Niveles de sensibilidad para template=true
    st.markdown("#### 📊 Niveles de sensibilidad:")
    
    # Datos fijos para template=true
    template_true_data = {
        'Nivel': ['template'],
        'Cantidad': [107263],
        'Porcentaje': [100.0]
    }
    
    # Crear gráfico de barras
    fig_sens_template = px.bar(
        x=template_true_data['Nivel'],
        y=template_true_data['Cantidad'],
        title="Distribución de Sensibilidad - Template TRUE",
        labels={'x': 'Nivel de Sensibilidad', 'y': 'Cantidad'},
        color=template_true_data['Cantidad'],
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_sens_template, use_container_width=True)
    
    # Mostrar tabla de conteos
    sens_df = pd.DataFrame(template_true_data)
    st.dataframe(sens_df, use_container_width=True)
    
    # Top documentos más comunes para template=true
    st.markdown("#### 📄 Top documentos más comunes:")
    
    # Datos fijos de documentos template=true
    docs_template_true = {
        'Documento': ['rpa signed', 'oficio nro urr signed', 'vista previa', 'solucion movimientos cuenta', 'consultar export'],
        'Cantidad': [41533, 5905, 4692, 4341, 2150],
        'Porcentaje': [38.7, 5.5, 4.4, 4.0, 2.0]
    }
    
    # Crear gráfico de barras horizontal
    fig_docs_template = px.bar(
        x=docs_template_true['Cantidad'],
        y=docs_template_true['Documento'],
        orientation='h',
        title="Top 5 Documentos Más Comunes - Template TRUE",
        labels={'x': 'Cantidad', 'y': 'Documento'}
    )
    fig_docs_template.update_layout(height=400)
    st.plotly_chart(fig_docs_template, use_container_width=True)
    
    # Mostrar tabla
    docs_df = pd.DataFrame(docs_template_true)
    st.dataframe(docs_df, use_container_width=True)
    
    # Análisis del documento más común
    st.markdown("#### 🔐 Análisis del documento más común: 'rpa signed'")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tiene Datos Personales", "Sí")
    with col2:
        st.metric("Cantidad", "41,533")
    with col3:
        st.metric("% del Template TRUE", "38.7%")
    
    st.markdown("---")
    
    # Análisis TEMPLATE = FALSE (datos fijos)
    st.subheader("🔍 Análisis para TEMPLATE = FALSE")
    
    # Calcular total de template=false
    total_template_false = 72250 + 16617 + 7457 + 3547
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documentos No-Template", f"{total_template_false:,}")
    
    with col2:
        pct_no_template = (total_template_false / 1353568) * 100
        st.metric("% del Total (Relabeling)", f"{pct_no_template:.2f}%")
    
    with col3:
        st.metric("Niveles de Sensibilidad", "4 tipos")
    
    # Análisis de sensibilidad para template=false
    st.markdown("#### 📊 Niveles de sensibilidad:")
    
    # Datos fijos para template=false
    template_false_data = {
        'Nivel': ['risk', 'sensitivity', 'regex', 'template'],
        'Cantidad': [72250, 16617, 7457, 3547],
        'Porcentaje': [72.2, 16.6, 7.4, 3.5]
    }
    
    # Crear gráfico de barras
    fig_sens_no_template = px.bar(
        x=template_false_data['Nivel'],
        y=template_false_data['Cantidad'],
        title="Distribución de Sensibilidad - Template FALSE",
        labels={'x': 'Nivel de Sensibilidad', 'y': 'Cantidad'},
        color=template_false_data['Cantidad'],
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_sens_no_template, use_container_width=True)
    
    # Mostrar tabla de conteos
    sens_false_df = pd.DataFrame(template_false_data)
    st.dataframe(sens_false_df, use_container_width=True)
    
    # Top documentos más comunes para template=false
    st.markdown("#### 📄 Top documentos más comunes:")
    
    # Datos fijos de documentos template=false
    docs_template_false = {
        'Documento': ['estado cuenta', 'cuadre boveda', 'vista previa', 'cheques devueltos', 'imp renta'],
        'Cantidad': [3035, 2616, 2114, 1508, 1501],
        'Porcentaje': [2.8, 2.4, 1.9, 1.4, 1.4]
    }
    
    # Crear gráfico de barras horizontal
    fig_docs_no_template = px.bar(
        x=docs_template_false['Cantidad'],
        y=docs_template_false['Documento'],
        orientation='h',
        title="Top 5 Documentos Más Comunes - Template FALSE",
        labels={'x': 'Cantidad', 'y': 'Documento'}
    )
    fig_docs_no_template.update_layout(height=400)
    st.plotly_chart(fig_docs_no_template, use_container_width=True)
    
    # Mostrar tabla
    docs_false_df = pd.DataFrame(docs_template_false)
    st.dataframe(docs_false_df, use_container_width=True)
    
    # Análisis del documento más común
    st.markdown("#### 🔐 Análisis del documento más común: 'estado cuenta'")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tiene Datos Personales", "Sí")
    with col2:
        st.metric("Cantidad", "3,035")
    with col3:
        st.metric("% del Template FALSE", "2.8%")
    
    # Comparación lado a lado
    st.markdown("---")
    st.subheader("📊 Comparación Template vs No-Template")
    
    # Métricas comparativas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Template TRUE", 
            "107,263",
            help="Documentos identificados como template"
        )
    
    with col2:
        st.metric(
            "Template FALSE", 
            f"{total_template_false:,}",
            help="Documentos NO identificados como template"
        )
    
    with col3:
        ratio = 107263 / total_template_false
        st.metric(
            "Ratio Template/No-Template", 
            f"{ratio:.2f}",
            help="Proporción entre documentos template y no-template"
        )
    
    # Gráfico comparativo de sensibilidad
    st.markdown("#### Comparación de Niveles de Sensibilidad")
    
    # Crear DataFrame para comparación
    comparison_data = {
        'Nivel': ['template (TRUE)', 'risk (FALSE)', 'sensitivity (FALSE)', 'regex (FALSE)', 'template (FALSE)'],
        'Cantidad': [107263, 72250, 16617, 7457, 3547],
        'Tipo': ['Template TRUE', 'Template FALSE', 'Template FALSE', 'Template FALSE', 'Template FALSE']
    }
    
    fig_comparison = px.bar(
        x=comparison_data['Nivel'],
        y=comparison_data['Cantidad'],
        color=comparison_data['Tipo'],
        title='Comparación Completa: Template TRUE vs FALSE por Nivel de Sensibilidad',
        labels={'x': 'Nivel de Sensibilidad', 'y': 'Cantidad'},
        color_discrete_map={'Template TRUE': 'lightblue', 'Template FALSE': 'lightcoral'}
    )
    fig_comparison.update_layout(height=500)
    st.plotly_chart(fig_comparison, use_container_width=True)

# SECCIÓN: EXPLORAR DATOS
elif section == "📊 Explorar Datos":
    st.header("Exploración de Dataset")
    
    if sensitivity_data is not None:
        
        current_df = sensitivity_data
        
        # Información general del dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Filas", f"{len(current_df):,}")
        
        with col2:
            st.metric("Total Columnas", len(current_df.columns))
        
        with col3:
            memory_usage = current_df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memoria (MB)", f"{memory_usage:.1f}")
        
        with col4:
            null_count = current_df.isnull().sum().sum()
            st.metric("Valores Nulos", f"{null_count:,}")
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Vista General", "📊 Distribuciones", "🎯 Filtros", "📋 Datos Raw"])
        
        with tab1:
            st.subheader("Información de Columnas")
            
            # Info de tipos de datos
            col_info = pd.DataFrame({
                'Columna': current_df.columns,
                'Tipo': current_df.dtypes,
                'Valores Únicos': [current_df[col].nunique() for col in current_df.columns],
                'Valores Nulos': [current_df[col].isnull().sum() for col in current_df.columns],
                '% Nulos': [round(current_df[col].isnull().sum() / len(current_df) * 100, 2) for col in current_df.columns]
            })
            
            st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            st.subheader("Distribuciones Clave")
            
            # Distribución por agente
            if 'data_source' in current_df.columns:
                fig_agent_dist = px.pie(
                    values=current_df['data_source'].value_counts().values,
                    names=current_df['data_source'].value_counts().index,
                    title="Distribución por Data Source"
                )
                st.plotly_chart(fig_agent_dist, use_container_width=True)
            
            # Distribución por clasificaciones
            if 'classifications' in current_df.columns:
                class_counts = current_df['classifications'].value_counts()
                fig_class = px.bar(
                    x=class_counts.index,
                    y=class_counts.values,
                    title="Distribución por Clasificaciones",
                    labels={'x': 'Clasificación', 'y': 'Cantidad'}
                )
                st.plotly_chart(fig_class, use_container_width=True)
            
            # Distribución por confidentiality model
            if 'confidentiality_model_obtained' in current_df.columns:
                conf_counts = current_df['confidentiality_model_obtained'].value_counts()
                fig_conf = px.bar(
                    x=conf_counts.index,
                    y=conf_counts.values,
                    title="Distribución por Confidentiality Model",
                    labels={'x': 'Modelo', 'y': 'Cantidad'}
                )
                st.plotly_chart(fig_conf, use_container_width=True)
        
        with tab3:
            st.subheader("Filtros Interactivos")
            
            # Crear filtros dinámicos
            col1, col2 = st.columns(2)
            
            with col1:
                # Filtro por data_source
                if 'data_source' in current_df.columns:
                    selected_sources = st.multiselect(
                        "Data Source:",
                        options=current_df['data_source'].unique(),
                        default=current_df['data_source'].unique()
                    )
                else:
                    selected_sources = None
                
                # Filtro por clasificación
                if 'classifications' in current_df.columns:
                    selected_classes = st.multiselect(
                        "Clasificaciones:",
                        options=sorted(current_df['classifications'].unique()),
                        default=sorted(current_df['classifications'].unique())[:5]  # Solo primeras 5
                    )
                else:
                    selected_classes = None
            
            with col2:
                # Filtro por confidentiality model
                if 'confidentiality_model_obtained' in current_df.columns:
                    selected_conf = st.multiselect(
                        "Confidentiality Model:",
                        options=current_df['confidentiality_model_obtained'].unique(),
                        default=current_df['confidentiality_model_obtained'].unique()
                    )
                else:
                    selected_conf = None
                
                # Filtro por vectores vacíos
                if 'empty_vector' in current_df.columns:
                    exclude_empty = st.checkbox("Excluir vectores vacíos", value=True)
                else:
                    exclude_empty = False
            
            # Aplicar filtros
            filtered_df = current_df.copy()
            
            if selected_sources and 'data_source' in current_df.columns:
                filtered_df = filtered_df[filtered_df['data_source'].isin(selected_sources)]
            
            if selected_classes and 'classifications' in current_df.columns:
                filtered_df = filtered_df[filtered_df['classifications'].isin(selected_classes)]
            
            if selected_conf and 'confidentiality_model_obtained' in current_df.columns:
                filtered_df = filtered_df[filtered_df['confidentiality_model_obtained'].isin(selected_conf)]
            
            if exclude_empty and 'empty_vector' in current_df.columns:
                filtered_df = filtered_df[filtered_df['empty_vector'] != True]
            
            st.info(f"Datos filtrados: {len(filtered_df):,} filas de {len(current_df):,} originales")
            
            # Mostrar accuracy de datos filtrados
            if len(filtered_df) > 0 and 'REAL' in filtered_df.columns and 'classifications' in filtered_df.columns:
                try:
                    from sklearn.metrics import confusion_matrix
                    
                    # Filtrar valores válidos (sin NaN)
                    valid_mask = ~(pd.isna(filtered_df['REAL']) | pd.isna(filtered_df['classifications']))
                    valid_data = filtered_df[valid_mask]
                    
                    if len(valid_data) > 0:
                        # Usar el mismo método que tu código
                        confusion = confusion_matrix(valid_data['REAL'], valid_data['classifications'])
                        accuracy_per_class = np.diag(confusion) / np.sum(confusion, axis=1)
                        
                        # Filtrar NaN y calcular promedio solo de valores válidos
                        valid_accuracies = accuracy_per_class[~np.isnan(accuracy_per_class)]
                        avg_accuracy = np.mean(valid_accuracies) if len(valid_accuracies) > 0 else 0.0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy Filtrado", f"{avg_accuracy:.4f}")
                        
                        with col2:
                            st.metric("Samples Válidos", f"{len(valid_data):,}")
                        
                        with col3:
                            st.metric("Clases Válidas", f"{len(valid_accuracies)}/{len(accuracy_per_class)}")
                        
                        # Mostrar accuracy por clase
                        st.subheader("Accuracy por Clase")
                        acc_per_class_df = pd.DataFrame({
                            'Clase': [f'Clase {i}' for i in range(len(accuracy_per_class))],
                            'Accuracy': accuracy_per_class,
                            'Estado': ['Válida' if not np.isnan(acc) else 'NaN' for acc in accuracy_per_class]
                        })
                        st.dataframe(acc_per_class_df, use_container_width=True)
                        
                        # Matriz de confusión de datos filtrados
                        if confusion.size > 0:
                            # Reemplazar NaN con 0 para visualización
                            confusion_pct = (confusion / confusion.sum(axis=1)[:, np.newaxis]) * 100
                            confusion_pct = np.nan_to_num(confusion_pct, nan=0.0)
                            fig_filtered = create_confusion_heatmap(confusion_pct, "Confusion Matrix - Datos Filtrados")
                            st.plotly_chart(fig_filtered, use_container_width=True)
                    else:
                        st.warning("No hay datos válidos después de filtrar NaN")
                
                except Exception as e:
                    st.warning(f"No se pudo calcular accuracy: {e}")
        
        with tab4:
            st.subheader("Vista de Datos Raw")
            
            # Selector de columnas a mostrar
            columns_to_show = st.multiselect(
                "Selecciona columnas:",
                options=current_df.columns.tolist(),
                default=['user', 'agent', 'classifications', 'REAL', 'data_source'][:5]
            )
            
            if columns_to_show:
                # Mostrar datos con paginación
                rows_per_page = st.slider("Filas por página:", 10, 1000, 100)
                
                total_rows = len(current_df)
                total_pages = (total_rows - 1) // rows_per_page + 1
                
                page = st.number_input("Página:", 1, total_pages, 1)
                
                start_idx = (page - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, total_rows)
                
                st.dataframe(
                    current_df[columns_to_show].iloc[start_idx:end_idx],
                    use_container_width=True
                )
                
                st.info(f"Mostrando filas {start_idx + 1} a {end_idx} de {total_rows}")
    
    else:
        st.error("No se pudieron cargar los datos. Verifica que el archivo ml_sensitivity.parquet esté en la carpeta 'data/'")

# Footer
st.markdown("---")
st.markdown("**Dashboard ML Accuracy** - Análisis completo de rendimiento de modelos")
st.markdown("*Datos actualizados: Junio 2025*")