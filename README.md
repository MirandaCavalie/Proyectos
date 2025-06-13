# 📊 Dashboard de Templates - BigQuery

Dashboard interactivo para análisis de templates con conexión segura a BigQuery.

## 🚀 Despliegue

### Streamlit Cloud
1. Sube el código a GitHub (sin credenciales)
2. Conecta tu repo en [share.streamlit.io](https://share.streamlit.io)
3. Configura secrets en la app:
   - Ve a Settings → Secrets
   - Copia el contenido de `.streamlit/secrets.toml`

### Variables de Entorno (Otras plataformas)
Configura estas variables:
- `GCP_PROJECT_ID`
- `GCP_PRIVATE_KEY_ID`
- `GCP_PRIVATE_KEY`
- `GCP_CLIENT_EMAIL`
- `GCP_CLIENT_ID`
- `GCP_CLIENT_X509_CERT_URL`

## 🔧 Instalación Local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🔐 Configuración de Credenciales

### Opción 1: Streamlit Secrets (Recomendado)
- Crea `.streamlit/secrets.toml` (no subir a GitHub)
- Copia datos de tu JSON de Google Cloud

### Opción 2: Variables de Entorno
- Configura las variables GCP_* en tu sistema

### Opción 3: Archivo Local (Solo desarrollo)
- Coloca `kriptos-credentials.json` en la raíz del proyecto

## 📊 Características

- 📈 Análisis temporal de templates
- 🔢 Análisis de contadores (tarjetas de crédito, PII)
- 📋 Distribución por clasificaciones
- 📄 Análisis de nombres de documentos
- 🔍 Exploración interactiva de datos

## 🛡️ Seguridad

- ✅ Credenciales nunca en el código
- ✅ Archivo JSON en .gitignore
- ✅ Múltiples métodos de autenticación
- ✅ Configuración segura para producción