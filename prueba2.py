import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Dashboard - Análisis por Ambiente",
    page_icon="🌐",
    layout="wide"
)

# Título principal
st.title("🌐 Dashboard - Análisis de Documentos por Ambiente")
st.markdown("---")

# Función para cargar datos
@st.cache_data
def cargar_datos():
    """Carga todos los archivos de datos generados"""
    try:
        # Cargar datos principales
        with open('resultados_analisis/analisis_completo.json', 'r', encoding='utf-8') as f:
            datos_completos = json.load(f)
        
        # Cargar datos para gráficos
        with open('resultados_analisis/datos_graficos.json', 'r', encoding='utf-8') as f:
            datos_graficos = json.load(f)
        
        # Cargar CSVs
        comp_6m = pd.read_csv('resultados_analisis/comparativo_6_meses.csv')
        comp_12m = pd.read_csv('resultados_analisis/comparativo_12_meses.csv')
        
        return datos_completos, datos_graficos, comp_6m, comp_12m
        
    except FileNotFoundError as e:
        st.error(f"❌ Error al cargar datos: {e}")
        st.info("📋 Ejecuta primero: python analisis_documentos.py")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        return None, None, None, None

# Cargar datos
resultado_carga = cargar_datos()

# Verificar que se cargaron los datos correctamente
if resultado_carga[0] is None:
    st.stop()

# Desempaquetar datos
datos_completos, datos_graficos, comp_6m, comp_12m = resultado_carga

# Sidebar con información general
st.sidebar.header("📊 Información General")
resumen = datos_completos['resumen_general']
st.sidebar.metric("Total Documentos", f"{resumen['total_documentos_analizados']:,}")
st.sidebar.metric("Total Ambientes", f"{resumen['total_ambientes']}")
st.sidebar.write(f"**Período:** {resumen['fecha_min']} a {resumen['fecha_max']}")
st.sidebar.write(f"**Generado:** {datos_completos['fecha_generacion']}")

# Selector de período
periodo_seleccionado = st.sidebar.selectbox(
    "Seleccionar Período:",
    ["6 meses", "12 meses", "Comparativo"]
)

# SECCIÓN 1: MÉTRICAS PRINCIPALES
st.header("📈 Métricas Principales")

# Crear columnas para métricas
col1, col2, col3, col4 = st.columns(4)

# Seleccionar datos según el período
if periodo_seleccionado == "6 meses":
    datos_periodo = datos_completos['resultados_6_meses']
    comp_periodo = comp_6m
elif periodo_seleccionado == "12 meses":
    datos_periodo = datos_completos['resultados_12_meses']
    comp_periodo = comp_12m
else:  # Comparativo
    datos_periodo = datos_completos['resultados_12_meses']  # Por defecto
    comp_periodo = comp_12m

# Mostrar métricas si no es comparativo
if periodo_seleccionado != "Comparativo":
    with col1:
        st.metric("Total Modificaciones", f"{datos_periodo['total_documentos']:,}")
    
    with col2:
        st.metric("Período Analizado", f"{datos_periodo['periodo']}")
    
    with col3:
        ambientes_activos = len(datos_periodo['analisis_por_ambiente'])
        st.metric("Ambientes Activos", f"{ambientes_activos}")
    
    with col4:
        if len(comp_periodo) > 0:
            ambiente_top = comp_periodo.iloc[0]['ambiente']
            st.metric("Ambiente Líder", ambiente_top)

# SECCIÓN 2: GRÁFICOS PRINCIPALES
st.header("📊 Visualizaciones")

# Tabs para diferentes vistas
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking Ambientes", "📈 Distribución", "📅 Tendencias", "🔍 Detalles"])

with tab1:
    st.subheader("🏆 Ranking de Ambientes por Actividad")
    
    if periodo_seleccionado == "Comparativo":
        # Mostrar comparativo lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Últimos 6 Meses**")
            if len(comp_6m) > 0:
                fig_6m = px.bar(
                    comp_6m, 
                    x='modificaciones', 
                    y='ambiente',
                    title="Top Ambientes - 6 Meses",
                    color='porcentaje',
                    color_continuous_scale='viridis',
                    orientation='h'
                )
                fig_6m.update_layout(height=400)
                st.plotly_chart(fig_6m, use_container_width=True)
        
        with col2:
            st.write("**Últimos 12 Meses**")
            if len(comp_12m) > 0:
                fig_12m = px.bar(
                    comp_12m, 
                    x='modificaciones', 
                    y='ambiente',
                    title="Top Ambientes - 12 Meses",
                    color='porcentaje',
                    color_continuous_scale='plasma',
                    orientation='h'
                )
                fig_12m.update_layout(height=400)
                st.plotly_chart(fig_12m, use_container_width=True)
    else:
        # Mostrar período seleccionado
        if len(comp_periodo) > 0:
            fig = px.bar(
                comp_periodo, 
                x='modificaciones', 
                y='ambiente',
                title=f"Ranking de Ambientes - {periodo_seleccionado}",
                color='porcentaje',
                color_continuous_scale='viridis',
                orientation='h',
                text='modificaciones'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=max(400, len(comp_periodo) * 40))
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📈 Distribución de Actividad")
    
    if periodo_seleccionado != "Comparativo" and len(comp_periodo) > 0:
        # Gráfico de pastel
        fig_pie = px.pie(
            comp_periodo,
            values='modificaciones',
            names='ambiente',
            title=f"Distribución de Modificaciones - {periodo_seleccionado}"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Gráfico de barras horizontal con detalles
        fig_bar = px.bar(
            comp_periodo,
            x='modificaciones',
            y='ambiente',
            title=f"Actividad por Ambiente - {periodo_seleccionado}",
            color='usuarios_activos',
            color_continuous_scale='blues',
            orientation='h'
        )
        fig_bar.update_layout(height=max(400, len(comp_periodo) * 50))
        st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.subheader("📅 Tendencias Mensuales")
    
    if 'tendencias_mensuales' in datos_graficos and datos_graficos['tendencias_mensuales']:
        # Selector de ambiente para tendencias
        ambientes_disponibles = list(datos_graficos['tendencias_mensuales'].keys())
        ambiente_seleccionado = st.selectbox("Seleccionar Ambiente:", ambientes_disponibles)
        
        if ambiente_seleccionado in datos_graficos['tendencias_mensuales']:
            tendencias = datos_graficos['tendencias_mensuales'][ambiente_seleccionado]
            
            # Preparar datos para el gráfico
            meses = list(tendencias.keys())
            modificaciones = [tendencias[mes]['modificaciones'] for mes in meses]
            usuarios = [tendencias[mes]['usuarios_activos'] for mes in meses]
            
            # Crear gráfico de líneas
            fig_tendencias = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Modificaciones por Mes', 'Usuarios Activos por Mes'),
                vertical_spacing=0.1
            )
            
            # Modificaciones
            fig_tendencias.add_trace(
                go.Scatter(
                    x=meses,
                    y=modificaciones,
                    mode='lines+markers',
                    name='Modificaciones',
                    line=dict(color='blue', width=3)
                ),
                row=1, col=1
            )
            
            # Usuarios
            fig_tendencias.add_trace(
                go.Scatter(
                    x=meses,
                    y=usuarios,
                    mode='lines+markers',
                    name='Usuarios Activos',
                    line=dict(color='red', width=3)
                ),
                row=2, col=1
            )
            
            fig_tendencias.update_layout(
                height=600,
                title_text=f"Tendencias Mensuales - {ambiente_seleccionado}",
                showlegend=False
            )
            fig_tendencias.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_tendencias, use_container_width=True)
        
        # Comparativa de todos los ambientes en un solo gráfico
        st.write("**Comparativa de Todos los Ambientes**")
        fig_todos = go.Figure()
        
        for ambiente, tendencias in datos_graficos['tendencias_mensuales'].items():
            meses = list(tendencias.keys())
            modificaciones = [tendencias[mes]['modificaciones'] for mes in meses]
            
            fig_todos.add_trace(
                go.Scatter(
                    x=meses,
                    y=modificaciones,
                    mode='lines+markers',
                    name=ambiente,
                    line=dict(width=2)
                )
            )
        
        fig_todos.update_layout(
            title="Tendencias Mensuales - Todos los Ambientes",
            xaxis_title="Mes",
            yaxis_title="Modificaciones",
            height=500
        )
        fig_todos.update_xaxes(tickangle=45)
        st.plotly_chart(fig_todos, use_container_width=True)

with tab4:
    st.subheader("🔍 Análisis Detallado por Ambiente")
    
    if periodo_seleccionado != "Comparativo":
        # Selector de ambiente para detalles
        ambientes_disponibles = list(datos_periodo['analisis_por_ambiente'].keys())
        ambiente_detalle = st.selectbox("Seleccionar Ambiente para Detalle:", ambientes_disponibles)
        
        if ambiente_detalle in datos_periodo['analisis_por_ambiente']:
            datos_ambiente = datos_periodo['analisis_por_ambiente'][ambiente_detalle]
            
            # Métricas del ambiente
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Modificaciones", f"{datos_ambiente['total_modificaciones']:,}")
            with col2:
                st.metric("Documentos Únicos", f"{datos_ambiente['documentos_unicos']:,}")
            with col3:
                st.metric("Usuarios Activos", f"{datos_ambiente['usuarios_activos']:,}")
            with col4:
                st.metric("% del Total", f"{datos_ambiente['porcentaje_del_total']:.1f}%")
            
            # Gráficos de top elementos
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Top Documentos**")
                if datos_ambiente['top_documentos']:
                    df_docs = pd.DataFrame(list(datos_ambiente['top_documentos'].items()), 
                                         columns=['Documento', 'Modificaciones'])
                    fig_docs = px.bar(df_docs, x='Modificaciones', y='Documento', 
                                    orientation='h', title="Top Tipos de Documentos")
                    st.plotly_chart(fig_docs, use_container_width=True)
            
            with col2:
                st.write("**Top Áreas**")
                if datos_ambiente['top_areas']:
                    df_areas = pd.DataFrame(list(datos_ambiente['top_areas'].items()), 
                                          columns=['Área', 'Modificaciones'])
                    fig_areas = px.bar(df_areas, x='Modificaciones', y='Área', 
                                     orientation='h', title="Top Áreas")
                    st.plotly_chart(fig_areas, use_container_width=True)
            
            with col3:
                st.write("**Top Usuarios**")
                if datos_ambiente['top_usuarios']:
                    df_usuarios = pd.DataFrame(list(datos_ambiente['top_usuarios'].items()), 
                                             columns=['Usuario', 'Modificaciones'])
                    fig_usuarios = px.bar(df_usuarios, x='Modificaciones', y='Usuario', 
                                        orientation='h', title="Top Usuarios")
                    st.plotly_chart(fig_usuarios, use_container_width=True)

# SECCIÓN 3: TABLAS DETALLADAS
st.header("📋 Tablas Detalladas")

if periodo_seleccionado == "Comparativo":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Últimos 6 Meses")
        if len(comp_6m) > 0:
            st.dataframe(comp_6m, use_container_width=True)
    
    with col2:
        st.subheader("Últimos 12 Meses")
        if len(comp_12m) > 0:
            st.dataframe(comp_12m, use_container_width=True)
else:
    st.subheader(f"Ranking Detallado - {periodo_seleccionado}")
    if len(comp_periodo) > 0:
        st.dataframe(comp_periodo, use_container_width=True)

# SECCIÓN 4: EXPORTAR DATOS
st.header("💾 Exportar Datos")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Descargar Resumen Ejecutivo"):
        try:
            with open('resultados_analisis/resumen_ejecutivo.txt', 'r', encoding='utf-8') as f:
                resumen_texto = f.read()
            st.download_button(
                label="📄 Descargar TXT",
                data=resumen_texto,
                file_name=f"resumen_ejecutivo_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        except FileNotFoundError:
            st.error("Archivo de resumen no encontrado")

with col2:
    if len(comp_periodo) > 0:
        csv_data = comp_periodo.to_csv(index=False)
        st.download_button(
            label="📊 Descargar CSV",
            data=csv_data,
            file_name=f"comparativo_{periodo_seleccionado.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("📋 Descargar JSON Completo"):
        json_data = json.dumps(datos_completos, ensure_ascii=False, indent=2)
        st.download_button(
            label="📋 Descargar JSON",
            data=json_data,
            file_name=f"analisis_completo_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("🌐 Dashboard de Análisis por Ambiente - Generado automáticamente")