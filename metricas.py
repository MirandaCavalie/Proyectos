import pandas as pd
import json
from datetime import datetime
import os

# Configuración
PARQUET_FILE = "BP_Templates_bigquery_ready.parquet"

def load_data():
    """Carga datos desde archivo Parquet"""
    print("🔄 Cargando datos desde Parquet...")
    
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(f"No se encontró el archivo: {PARQUET_FILE}")
    
    df = pd.read_parquet(PARQUET_FILE)
    print(f"✅ Datos cargados: {len(df):,} filas, {len(df.columns)} columnas")
    
    return df

def process_data(df):
    """Procesa los datos y extrae métricas"""
    print("🔧 Procesando datos...")
    
    # Procesar fechas
    if 'modif_date' in df.columns:
        df['modif_date'] = pd.to_datetime(df['modif_date'], errors='coerce', utc=True)
        if df['modif_date'].dt.tz is not None:
            df['modif_date'] = df['modif_date'].dt.tz_convert(None)
        df['year_month'] = df['modif_date'].dt.to_period('M').astype(str)
        df['year'] = df['modif_date'].dt.year
        df['month'] = df['modif_date'].dt.month
    
    # Limpiar datos
    if 'doc_name' in df.columns:
        df['doc_name_clean'] = df['doc_name'].fillna('Sin nombre')
    
    if 'classifications' in df.columns:
        df['classifications_clean'] = df['classifications'].fillna('Sin clasificación')
    
    # Identificar columnas de contadores
    counter_columns = [col for col in df.columns if 'counter' in str(col).lower() or '_count' in col.lower()]
    
    print(f"📊 Columnas encontradas: {len(df.columns)}")
    print(f"🔢 Contadores detectados: {len(counter_columns)}")
    
    return df, counter_columns

def calculate_metrics(df, counter_columns):
    """Calcula todas las métricas del dashboard"""
    print("📈 Calculando métricas...")
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'source': 'parquet_file'
    }
    
    # 1. Métricas principales
    metrics['main_metrics'] = {
        'total_templates': len(df),
        'unique_classifications': df['classifications_clean'].nunique() if 'classifications_clean' in df.columns else 0,
        'counter_columns_detected': len(counter_columns),
        'date_range': {}
    }
    
    # Rango de fechas
    if 'modif_date' in df.columns:
        valid_dates = df['modif_date'].dropna()
        if len(valid_dates) > 0:
            metrics['main_metrics']['date_range'] = {
                'min_date': valid_dates.min().strftime('%Y-%m-%d'),
                'max_date': valid_dates.max().strftime('%Y-%m-%d'),
                'valid_dates_count': len(valid_dates)
            }
    
    # 2. Distribución por clasificación
    if 'classifications_clean' in df.columns:
        class_counts = df['classifications_clean'].value_counts()
        metrics['classification_distribution'] = {
            'data': class_counts.to_dict(),
            'top_15': class_counts.head(15).to_dict()
        }
        
        # Análisis detallado por clasificación 1, 2, 3
        detailed_classifications = {}
        for classification in ['1', '2', '3']:
            if classification in df['classifications_clean'].values:
                class_data = df[df['classifications_clean'] == classification]
                
                # Sumar contadores para esta clasificación
                total_counters = 0
                for col in counter_columns:
                    if col in class_data.columns and class_data[col].dtype in ['int64', 'float64']:
                        total_counters += class_data[col].sum()
                
                detailed_classifications[classification] = {
                    'count': len(class_data),
                    'percentage': (len(class_data) / len(df)) * 100,
                    'total_counters': int(total_counters),
                    'unique_docs': class_data['doc_name_clean'].nunique() if 'doc_name_clean' in class_data.columns else 0
                }
        
        metrics['detailed_classifications'] = detailed_classifications
    
    # 3. Análisis de contadores
    counter_stats = []
    for col in counter_columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            non_zero_count = (df[col] > 0).sum()
            percentage = (non_zero_count / len(df)) * 100
            
            counter_stats.append({
                'counter_name': col,
                'clean_name': col.replace('_count', '').replace('_', ' ').title(),
                'documents_with_data': int(non_zero_count),
                'percentage': round(percentage, 2),
                'total_sum': int(df[col].sum()),
                'max_value': int(df[col].max()),
                'average': round(df[col].mean(), 2)
            })
    
    metrics['counter_analysis'] = {
        'general_stats': sorted(counter_stats, key=lambda x: x['percentage'], reverse=True),
        'specific_categories': {}
    }
    
    # Contadores específicos por categoría
    specific_counters = {
        'credit_cards': ['cc_discover_count', 'cc_visa_count', 'cc_diners_club_count', 'cc_mastercard_count'],
        'pii_data': ['pii_address_count', 'pii_ruc_ecu_count', 'pii_ced_ecu_count', 
                     'pii_phone_number_count', 'pii_personal_name_count', 'pii_email_count']
    }
    
    for category, counters in specific_counters.items():
        existing_counters = [col for col in counters if col in df.columns]
        category_stats = []
        
        for counter in existing_counters:
            if df[counter].dtype in ['int64', 'float64']:
                docs_with_data = (df[counter] > 0).sum()
                percentage = (docs_with_data / len(df)) * 100
                total_count = df[counter].sum()
                
                category_stats.append({
                    'counter_name': counter,
                    'clean_name': counter.replace('_count', '').replace('_', ' ').title(),
                    'total_occurrences': int(total_count),
                    'documents_with_data': int(docs_with_data),
                    'percentage': round(percentage, 2)
                })
        
        metrics['counter_analysis']['specific_categories'][category] = category_stats
    
    # 4. Análisis temporal
    if 'modif_date' in df.columns:
        df_temporal = df[df['modif_date'].notna()]
        if len(df_temporal) > 0:
            # Por mes
            monthly_counts = df_temporal.groupby('year_month').size()
            monthly_data = []
            for month, count in monthly_counts.items():
                monthly_data.append({
                    'month': str(month),
                    'count': int(count)
                })
            
            # Por año
            yearly_counts = df_temporal.groupby('year').size()
            yearly_data = []
            for year, count in yearly_counts.items():
                yearly_data.append({
                    'year': int(year),
                    'count': int(count)
                })
            
            # Por mes del año (1-12)
            month_counts = df_temporal.groupby('month').size()
            month_data = []
            for month, count in month_counts.items():
                month_data.append({
                    'month_number': int(month),
                    'count': int(count)
                })
            
            metrics['temporal_analysis'] = {
                'monthly_data': sorted(monthly_data, key=lambda x: x['month']),
                'yearly_data': sorted(yearly_data, key=lambda x: x['year']),
                'month_distribution': sorted(month_data, key=lambda x: x['month_number'])
            }
    
    # 5. Análisis de nombres de documentos
    if 'doc_name_clean' in df.columns:
        name_counts = df['doc_name_clean'].value_counts()
        
        # Top 20 nombres más comunes
        top_names = []
        for name, count in name_counts.head(20).items():
            top_names.append({
                'document_name': str(name),
                'frequency': int(count)
            })
        
        # Estadísticas generales
        doc_stats = {
            'total_documents': len(df),
            'unique_names': df['doc_name_clean'].nunique(),
            'top_names': top_names
        }
        
        # Análisis de extensiones
        try:
            extensions = df['doc_name_clean'].str.extract(r'\.([^.]+)$')[0].value_counts()
            extension_data = []
            for ext, count in extensions.items():
                if pd.notna(ext):
                    extension_data.append({
                        'extension': str(ext),
                        'count': int(count)
                    })
            
            doc_stats['extensions'] = extension_data
            if extension_data:
                doc_stats['most_common_extension'] = extension_data[0]
        except:
            doc_stats['extensions'] = []
        
        metrics['document_analysis'] = doc_stats
    
    # 6. Estadísticas generales del dataset
    metrics['dataset_info'] = {
        'shape': {
            'rows': df.shape[0],
            'columns': df.shape[1]
        },
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'data_types': df.dtypes.astype(str).value_counts().to_dict(),
        'null_values': {
            'total_nulls': int(df.isnull().sum().sum()),
            'columns_with_most_nulls': {str(k): int(v) for k, v in df.isnull().sum().nlargest(5).items()}
        }
    }
    
    # 7. Información de columnas
    column_info = []
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
            'unique_values': int(df[col].nunique())
        }
        
        # Agregar estadísticas para columnas numéricas
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
                'max': float(df[col].max()) if pd.notna(df[col].max()) else None,
                'mean': round(float(df[col].mean()), 2) if pd.notna(df[col].mean()) else None
            })
        
        column_info.append(col_info)
    
    metrics['column_info'] = column_info
    
    # 8. Muestra de datos (primeras 100 filas para demo)
    sample_data = df.head(100).copy()
    
    # Convertir fechas a string para JSON
    for col in sample_data.columns:
        if pd.api.types.is_datetime64_any_dtype(sample_data[col]):
            sample_data[col] = sample_data[col].astype(str)
    
    # Convertir otros tipos problemáticos
    for col in sample_data.columns:
        if sample_data[col].dtype == 'object':
            sample_data[col] = sample_data[col].astype(str)
    
    metrics['sample_data'] = {
        'columns': list(sample_data.columns),
        'data': sample_data.to_dict('records')
    }
    
    print("✅ Métricas calculadas exitosamente!")
    return metrics

def save_metrics(metrics):
    """Guarda las métricas en archivos"""
    
    # Crear carpeta de salida
    output_dir = "dashboard_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar métricas completas en JSON
    json_file = os.path.join(output_dir, "dashboard_metrics.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Guardar resumen en texto
    summary_file = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== RESUMEN DE MÉTRICAS DEL DASHBOARD ===\n")
        f.write(f"Generado: {metrics['timestamp']}\n")
        f.write(f"Fuente: {PARQUET_FILE}\n\n")
        
        f.write("MÉTRICAS PRINCIPALES:\n")
        main = metrics['main_metrics']
        f.write(f"- Total Templates: {main['total_templates']:,}\n")
        f.write(f"- Clasificaciones Únicas: {main['unique_classifications']}\n")
        f.write(f"- Contadores Detectados: {main['counter_columns_detected']}\n")
        if main['date_range']:
            f.write(f"- Rango de Fechas: {main['date_range']['min_date']} - {main['date_range']['max_date']}\n")
        
        f.write("\nTOP 10 CLASIFICACIONES:\n")
        if 'classification_distribution' in metrics:
            for i, (classification, count) in enumerate(list(metrics['classification_distribution']['data'].items())[:10], 1):
                f.write(f"{i}. {classification}: {count:,} templates\n")
        
        f.write("\nTOP 10 CONTADORES:\n")
        if 'counter_analysis' in metrics:
            for i, counter in enumerate(metrics['counter_analysis']['general_stats'][:10], 1):
                f.write(f"{i}. {counter['clean_name']}: {counter['documents_with_data']:,} docs ({counter['percentage']:.1f}%)\n")
        
        f.write("\nCONTADORES DE TARJETAS DE CRÉDITO:\n")
        if 'counter_analysis' in metrics and 'credit_cards' in metrics['counter_analysis']['specific_categories']:
            for counter in metrics['counter_analysis']['specific_categories']['credit_cards']:
                f.write(f"- {counter['clean_name']}: {counter['total_occurrences']:,} ocurrencias en {counter['documents_with_data']:,} docs\n")
        
        f.write("\nCONTADORES DE INFORMACIÓN PERSONAL (PII):\n")
        if 'counter_analysis' in metrics and 'pii_data' in metrics['counter_analysis']['specific_categories']:
            for counter in metrics['counter_analysis']['specific_categories']['pii_data']:
                f.write(f"- {counter['clean_name']}: {counter['total_occurrences']:,} ocurrencias en {counter['documents_with_data']:,} docs\n")
        
        f.write("\nANÁLISIS DETALLADO POR CLASIFICACIÓN:\n")
        if 'detailed_classifications' in metrics:
            for classification, data in metrics['detailed_classifications'].items():
                f.write(f"Clasificación {classification}:\n")
                f.write(f"  - Templates: {data['count']:,} ({data['percentage']:.1f}%)\n")
                f.write(f"  - Total Contadores: {data['total_counters']:,}\n")
                f.write(f"  - Nombres Únicos: {data['unique_docs']:,}\n")
        
        f.write("\nINFORMACIÓN DEL DATASET:\n")
        dataset = metrics['dataset_info']
        f.write(f"- Dimensiones: {dataset['shape']['rows']:,} filas × {dataset['shape']['columns']} columnas\n")
        f.write(f"- Memoria: {dataset['memory_usage_mb']:.2f} MB\n")
        f.write(f"- Valores nulos: {dataset['null_values']['total_nulls']:,}\n")
        
        if 'temporal_analysis' in metrics:
            f.write(f"\nANÁLISIS TEMPORAL:\n")
            yearly_data = metrics['temporal_analysis']['yearly_data']
            if yearly_data:
                f.write("- Distribución por año:\n")
                for year_data in yearly_data[-5:]:  # Últimos 5 años
                    f.write(f"  {year_data['year']}: {year_data['count']:,} templates\n")
        
        if 'document_analysis' in metrics:
            f.write(f"\nANÁLISIS DE DOCUMENTOS:\n")
            doc_stats = metrics['document_analysis']
            f.write(f"- Total documentos: {doc_stats['total_documents']:,}\n")
            f.write(f"- Nombres únicos: {doc_stats['unique_names']:,}\n")
            if 'most_common_extension' in doc_stats:
                ext = doc_stats['most_common_extension']
                f.write(f"- Extensión más común: .{ext['extension']} ({ext['count']:,} archivos)\n")
    
    print(f"✅ Métricas guardadas en:")
    print(f"   📄 JSON completo: {json_file}")
    print(f"   📝 Resumen: {summary_file}")
    
    return json_file, summary_file

def main():
    """Función principal"""
    try:
        print("🚀 Iniciando cálculo de métricas del dashboard desde Parquet...")
        
        # Cargar datos
        df = load_data()
        
        # Mostrar información básica
        print(f"📊 Información del dataset:")
        print(f"   - Filas: {len(df):,}")
        print(f"   - Columnas: {len(df.columns)}")
        print(f"   - Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Procesar datos
        df, counter_columns = process_data(df)
        
        # Calcular métricas
        metrics = calculate_metrics(df, counter_columns)
        
        # Guardar métricas
        json_file, summary_file = save_metrics(metrics)
        
        print("\n🎉 ¡Proceso completado exitosamente!")
        print(f"Total de registros procesados: {len(df):,}")
        print(f"Contadores encontrados: {len(counter_columns)}")
        print(f"Columnas procesadas: {len(df.columns)}")
        
        # Mostrar vista previa de las métricas principales
        if 'main_metrics' in metrics:
            main = metrics['main_metrics']
            print(f"\n📊 VISTA PREVIA DE MÉTRICAS:")
            print(f"   📄 Total Templates: {main['total_templates']:,}")
            print(f"   🏷️ Clasificaciones Únicas: {main['unique_classifications']}")
            print(f"   🔢 Contadores Detectados: {main['counter_columns_detected']}")
            if main['date_range']:
                print(f"   📅 Rango de Fechas: {main['date_range']['min_date']} - {main['date_range']['max_date']}")
        
        return json_file, summary_file
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()