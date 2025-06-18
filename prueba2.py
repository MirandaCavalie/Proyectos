import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Reporte de Documentos Modificados para Evaluación de Impacto DLP",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔒 Reporte de Documentos Modificados para Evaluación de Impacto DLP")
st.markdown("---")

# Sidebar con filtros
st.sidebar.header("🔧 Configuración")
periodo_seleccionado = st.sidebar.selectbox(
    "Selecciona el período de análisis:",
    ["Últimos 6 meses", "Últimos 12 meses"]
)

# Datos principales actualizados
datos_generales = {
    'total_docs_validos': 8025497,
    'fechas_invalidas': 0,
    'docs_filtrados_futuras': 35,
    'usuarios_unicos_12m': 16328,
    'areas_unicas_12m': 781,
    'mes_mas_activo': '2024-10 (780,052 modificaciones)',
    'promedio_mensual': 617346
}

datos_6_meses = {
    'docs_modificados': 3840616,
    'porcentaje_total': 47.86,
    'categorias_unicas': 843303
}

datos_12_meses = {
    'docs_modificados': 8015802,
    'porcentaje_total': 99.88,
    'categorias_unicas': 1664332
}

# === ALERTA DE FILTROS APLICADOS ===
st.info("🔍 **Filtros aplicados:** Fechas <= 2025-06-30 | Fechas inválidas filtradas: 35 documentos")

# === SECCIÓN 1: MÉTRICAS PRINCIPALES ===
st.header("📈 Métricas Principales")

if periodo_seleccionado == "Últimos 6 meses":
    datos_actuales = datos_6_meses
    periodo_texto = "6M"
else:
    datos_actuales = datos_12_meses
    periodo_texto = "12M"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📄 Total Documentos Válidos",
        value=f"{datos_generales['total_docs_validos']:,}",
        delta=f"{datos_actuales['docs_modificados']:,} ({periodo_texto})"
    )

with col2:
    st.metric(
        label=f"📂 Categorías Únicas ({periodo_texto})",
        value=f"{datos_actuales['categorias_unicas']:,}",
        delta=f"{datos_actuales['porcentaje_total']:.1f}% del total"
    )

with col3:
    st.metric(
        label="👥 Usuarios Únicos",
        value=f"{datos_generales['usuarios_unicos_12m']:,}",
        delta="Últimos 12 meses"
    )

with col4:
    st.metric(
        label="🏢 Áreas Únicas", 
        value=f"{datos_generales['areas_unicas_12m']:,}",
        delta="781 departamentos"
    )

st.markdown("---")

# === SECCIÓN 2: TOP 20 DOCUMENTOS SEGÚN PERÍODO SELECCIONADO ===
st.header(f"📄 TOP 20 Documentos Más Modificados ({periodo_seleccionado})")

# Datos TOP 20 - 6 meses
top_20_6m = {
    'doc_name': ['log', 'xxxxxxxx', 'tarjeta registro firma', 'filtertrie intermediate', 
                 'uihistory log', 'debug dump', 'rpa signed', 'license', 
                 'solicitud servicios electronicos', 'iva', 'rol', 'notice', 'data', 
                 'factura', 'ruc', 'scanned lexmark multifunction product', 'cedula', 
                 'xxxxxxxx signed', 'uac', 'vista previa'],
    'num_areas': [111, 93, 26, 113, 346, 370, 11, 312, 29, 71, 406, 365, 285, 262, 134, 272, 223, 3, 38, 66],
    'num_usuarios': [3059, 1107, 933, 479, 6491, 7513, 16, 2416, 100, 875, 4429, 6279, 2255, 1209, 1424, 2310, 1816, 18, 434, 627],
    'num_documentos': [121932, 75243, 70775, 49250, 40610, 37319, 32136, 27219, 24626, 24241, 22781, 21670, 19175, 19123, 19039, 18181, 18112, 17883, 17781, 17242]
}

# Datos TOP 20 - 12 meses
top_20_12m = {
    'doc_name': ['log', 'tarjeta registro firma', 'xxxxxxxx', 'rpa signed', 
                 'solicitud servicios electronicos', 'iva', 'uihistory log', 'filtertrie intermediate',
                 'scanned lexmark multifunction product', 'license', 'rol', 'cedula', 'data',
                 'ruc', 'debug dump', 'getting started', 'xxxxxxxx signed', 'vista previa',
                 'factura', 'solucion movimientos cuenta'],
    'num_areas': [153, 35, 144, 12, 40, 91, 357, 113, 335, 331, 460, 286, 342, 166, 370, 129, 4, 72, 331, 46],
    'num_usuarios': [3356, 1099, 1748, 23, 163, 1156, 6837, 528, 3395, 2603, 5671, 2688, 3306, 1953, 7514, 1483, 19, 728, 1974, 1242],
    'num_documentos': [196149, 152774, 149828, 57098, 56901, 53169, 52019, 51999, 50759, 49939, 42431, 41113, 39957, 38969, 37322, 36570, 32975, 32292, 31104, 30064]
}

# Seleccionar datos según el período
if periodo_seleccionado == "Últimos 6 meses":
    df_top20 = pd.DataFrame(top_20_6m)
else:
    df_top20 = pd.DataFrame(top_20_12m)

# Tabs para diferentes vistas
tab1, tab2, tab3 = st.tabs(["📊 Por Documentos", "👥 Por Usuarios", "🏢 Por Áreas"])

with tab1:
    # Gráfico de barras - Documentos
    fig_docs = px.bar(
        df_top20.head(10), 
        x='num_documentos', 
        y='doc_name',
        title=f"TOP 10 Documentos por Número de Modificaciones ({periodo_seleccionado})",
        labels={'num_documentos': 'Número de Documentos', 'doc_name': 'Tipo de Documento'},
        color='num_documentos',
        color_continuous_scale='Blues'
    )
    fig_docs.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_docs, use_container_width=True)
    
    # Tabla detallada
    st.subheader(f"📋 Tabla Detallada - TOP 20 ({periodo_seleccionado})")
    st.dataframe(
        df_top20.style.format({
            'num_documentos': '{:,}',
            'num_usuarios': '{:,}',
            'num_areas': '{:,}'
        }),
        use_container_width=True
    )

with tab2:
    # Gráfico de barras - Usuarios
    fig_users = px.bar(
        df_top20.head(10), 
        x='num_usuarios', 
        y='doc_name',
        title=f"TOP 10 Documentos por Número de Usuarios Únicos ({periodo_seleccionado})",
        labels={'num_usuarios': 'Número de Usuarios', 'doc_name': 'Tipo de Documento'},
        color='num_usuarios',
        color_continuous_scale='Greens'
    )
    fig_users.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_users, use_container_width=True)

with tab3:
    # Gráfico de barras - Áreas
    fig_areas = px.bar(
        df_top20.head(10), 
        x='num_areas', 
        y='doc_name',
        title=f"TOP 10 Documentos por Número de Áreas ({periodo_seleccionado})",
        labels={'num_areas': 'Número de Áreas', 'doc_name': 'Tipo de Documento'},
        color='num_areas',
        color_continuous_scale='Reds'
    )
    fig_areas.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_areas, use_container_width=True)

st.markdown("---")

# === SECCIÓN 3: ANÁLISIS TEMPORAL - SOLO PARA 12 MESES ===
if periodo_seleccionado == "Últimos 12 meses":
    st.header("📅 Análisis Temporal - Últimos 12 Meses")

    # Datos mensuales actualizados
    meses = ['2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12', 
             '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06']
    modificaciones = [648491, 597470, 686385, 780052, 684073, 674432, 
                      652048, 694036, 652051, 736888, 761145, 105968]
    docs_unicos = [191020, 178788, 214465, 201643, 196615, 193038, 
                   186175, 199984, 166423, 184734, 198395, 42271]
    usuarios = [11452, 11166, 11637, 12306, 12527, 13002, 
                13563, 13512, 13579, 13691, 13797, 7220]

    df_temporal = pd.DataFrame({
        'mes': meses,
        'modificaciones': modificaciones,
        'docs_unicos': docs_unicos,
        'usuarios': usuarios
    })

    # Gráfico de líneas múltiples
    fig_temporal = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Modificaciones por Mes', 'Documentos Únicos por Mes', 
                       'Usuarios Activos por Mes', 'Tendencia Combinada'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )

    # Modificaciones
    fig_temporal.add_trace(
        go.Scatter(x=df_temporal['mes'], y=df_temporal['modificaciones'], 
                   mode='lines+markers', name='Modificaciones', line=dict(color='blue', width=3)),
        row=1, col=1
    )

    # Documentos únicos
    fig_temporal.add_trace(
        go.Scatter(x=df_temporal['mes'], y=df_temporal['docs_unicos'], 
                   mode='lines+markers', name='Docs Únicos', line=dict(color='green', width=3)),
        row=1, col=2
    )

    # Usuarios
    fig_temporal.add_trace(
        go.Scatter(x=df_temporal['mes'], y=df_temporal['usuarios'], 
                   mode='lines+markers', name='Usuarios', line=dict(color='red', width=3)),
        row=2, col=1
    )

    # Tendencia combinada
    fig_temporal.add_trace(
        go.Scatter(x=df_temporal['mes'], y=df_temporal['modificaciones'], 
                   mode='lines+markers', name='Modificaciones', line=dict(color='blue', width=3)),
        row=2, col=2
    )

    fig_temporal.update_layout(height=600, showlegend=True, title_text="Análisis Temporal Completo")
    fig_temporal.update_xaxes(tickangle=45)
    st.plotly_chart(fig_temporal, use_container_width=True)

    # Estadísticas temporales
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Estadísticas del Período")
        st.write(f"**Promedio mensual:** {datos_generales['promedio_mensual']:,} modificaciones")
        st.write(f"**Mes más activo:** {datos_generales['mes_mas_activo']}")
        st.write(f"**Mes menos activo:** Junio 2025 (105,968)*")
        st.caption("*Datos parciales de junio")

    with col2:
        st.subheader("📈 Tendencias Clave")
        st.write(f"**Crecimiento de usuarios:** +21% (Jul'24 vs May'25)")
        st.write(f"**Pico de actividad:** Octubre 2024")
        st.write(f"**Actividad Q1 2025:** {np.mean([652048, 694036, 652051]):,.0f} promedio")

    st.markdown("---")

# === SECCIÓN 4: COMPARACIÓN 6M vs 12M ===
st.header("📊 Comparación de Períodos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Últimos 6 Meses")
    st.metric("📄 Documentos Modificados", f"{datos_6_meses['docs_modificados']:,}")
    st.metric("📂 Categorías Únicas", f"{datos_6_meses['categorias_unicas']:,}")
    st.metric("📊 % del Total", f"{datos_6_meses['porcentaje_total']:.1f}%")

with col2:
    st.subheader("📈 Últimos 12 Meses") 
    st.metric("📄 Documentos Modificados", f"{datos_12_meses['docs_modificados']:,}")
    st.metric("📂 Categorías Únicas", f"{datos_12_meses['categorias_unicas']:,}")
    st.metric("📊 % del Total", f"{datos_12_meses['porcentaje_total']:.1f}%")

# Gráfico comparativo
fig_comparacion = go.Figure()

periodos = ['6 Meses', '12 Meses']
docs_modificados = [datos_6_meses['docs_modificados'], datos_12_meses['docs_modificados']]
categorias = [datos_6_meses['categorias_unicas'], datos_12_meses['categorias_unicas']]

fig_comparacion.add_trace(go.Bar(
    name='Documentos Modificados',
    x=periodos,
    y=docs_modificados,
    yaxis='y',
    offsetgroup=1,
    marker_color='lightblue'
))

fig_comparacion.add_trace(go.Bar(
    name='Categorías Únicas',
    x=periodos,
    y=categorias,
    yaxis='y2',
    offsetgroup=2,
    marker_color='lightcoral'
))

fig_comparacion.update_layout(
    title='Comparación: 6 Meses vs 12 Meses',
    xaxis=dict(title='Período'),
    yaxis=dict(title='Documentos Modificados', side='left'),
    yaxis2=dict(title='Categorías Únicas', side='right', overlaying='y'),
    barmode='group',
    height=400
)

st.plotly_chart(fig_comparacion, use_container_width=True)

# Insight clave
diferencia_docs = datos_12_meses['docs_modificados'] - datos_6_meses['docs_modificados']
porcentaje_6m_de_12m = (datos_6_meses['docs_modificados'] / datos_12_meses['docs_modificados']) * 100

st.info(f"""
**📊 Insight Clave:** 
- Los últimos 6 meses representan el **{porcentaje_6m_de_12m:.1f}%** de toda la actividad de 12 meses
- Diferencia: **{diferencia_docs:,}** documentos adicionales en 12M vs 6M
- Esto indica una **actividad sostenida** a lo largo del año
""")

st.markdown("---")

# === SECCIÓN 5: TOP ÁREAS POR PERÍODO ===
st.header(f"🏢 TOP 10 Áreas Más Activas ({periodo_seleccionado})")

# Datos de áreas por período
if periodo_seleccionado == "Últimos 6 meses":
    areas_data = {
        'area': ['EXTERNA', 'REGION COSTA', 'POR REVISAR AREA', 'REGION NORTE', 
                 'BANCA RELACIONAL', 'REGION CENTRO', 'BANCO PICHINCHA C.A.', 
                 'FABRICA OPERACIONES', 'COMERCIAL AUTOMOTRIZ', 'PASANTE'],
        'modificaciones': [324264, 306712, 267867, 265675, 141786, 136803, 110402, 77325, 74454, 73273]
    }
else:
    areas_data = {
        'area': ['REGION COSTA', 'EXTERNA', 'REGION NORTE', 'POR REVISAR AREA', 
                 'REGION CENTRO', 'COMERCIAL AUTOMOTRIZ', 'BANCO PICHINCHA C.A.', 
                 'BANCA RELACIONAL', 'FABRICA OPERACIONES', 'SERVICIO AL CLIENTE'],
        'modificaciones': [692814, 594770, 579077, 502132, 326016, 258158, 246959, 221224, 193781, 140770]
    }

df_areas = pd.DataFrame(areas_data)

fig_areas_pie = px.pie(
    df_areas, 
    values='modificaciones', 
    names='area',
    title=f"Distribución de Modificaciones por Área ({periodo_seleccionado})"
)
fig_areas_pie.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_areas_pie, use_container_width=True)

st.markdown("---")

# === FOOTER - RESUMEN EJECUTIVO ===
st.markdown("### 📋 Resumen Ejecutivo")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"""
    **📅 Período:** 2024-06-18 a 2025-06-18
    **📄 Total Documentos:** {datos_generales['total_docs_validos']:,}
    **🔍 Filtros:** 35 docs con fechas futuras
    """)

with col2:
    st.success(f"""
    **📊 Actividad {periodo_texto}:** {datos_actuales['docs_modificados']:,}
    **📂 Categorías:** {datos_actuales['categorias_unicas']:,}
    **📈 Cobertura:** {datos_actuales['porcentaje_total']:.1f}%
    """)

with col3:
    st.warning(f"""
    **🔥 Pico:** Octubre 2024 (780K)
    **📊 Promedio:** {datos_generales['promedio_mensual']:,}/mes
    **👥 Usuarios:** {datos_generales['usuarios_unicos_12m']:,} únicos
    """)

st.caption("Reporte DLP generado automáticamente • Datos actualizados a Junio 2025 • Filtros aplicados: fechas válidas <= 2025-06-30")