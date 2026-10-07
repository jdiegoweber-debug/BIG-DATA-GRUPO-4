
# =============================================================================
# CONSULTA 4 (Obligatoria para todos los grupos)
# =============================================================================
# CONSIGNA OFICIAL:
# Hotspots espaciales: top 5 puntos más peligrosos.
# Identificar y reportar las 5 coordenadas geográficas más peligrosas (a nivel nacional 
# o en un estado de alto flujo como California o Texas) dado un radio geográfico de 
# influencia (ej. R = 2.0 km o 5.0 km). El equipo debe definir formalmente el criterio 
# de peligrosidad (densidad absoluta, concentración de accidentes con Severity >= 3, 
# o índice ponderado) y justificar el método espacial implementado.
#
# JUSTIFICACIÓN METODOLÓGICA Y ESPACIAL DEL GRUPO 4:
# 1. Criterio de Peligrosidad: Concentración absoluta de accidentes con Severidad Alta (Severity >= 3).
#    Esto prioriza zonas de alto impacto y riesgo vial crítico, ignorando siniestros menores.
# 2. Método Espacial (Grilla por Truncamiento): Calcular distancias de Haversine continuas para 
#    7.7 millones de registros mononodo genera un cuello de botella de memoria (OOM) en Pandas. 
#    Para simular el radio de influencia (R ~ 2.0-5.0 km) de forma eficiente a gran escala, se 
#    implementa una discretización espacial redondeando las coordenadas a 2 decimales. 
#    Un cambio de 0.01 grados en latitud/longitud equivale aproximadamente a ~1.1 km, por lo que 
#    una celda de 2 decimales representa una región de influencia controlada de aprox. 2.2 km.
# =============================================================================

# =============================================================================
# GRUPO 4 - Consulta 4: Hotspots espaciales (Top 5 puntos más peligrosos)
# =============================================================================

import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n🚀 [PANDAS] Ejecutando Consulta 4...")
    start = time.time()
    
    # Lectura inteligente de columnas necesarias
    df = pd.read_parquet(path, columns=['Start_Lat', 'Start_Lng', 'Severity'])
    df_graves = df[(df['Severity'] >= 3) & (df['Start_Lat'].notna())].copy()
    
    # Grilla espacial por redondeo a 2 decimales (~2.2 km de influencia)
    df_graves['Lat_Grid'] = df_graves['Start_Lat'].round(2)
    df_graves['Lng_Grid'] = df_graves['Start_Lng'].round(2)
    
    top_5 = df_graves.groupby(['Lat_Grid', 'Lng_Grid']).size().reset_index(name='Accidentes').sort_values(by='Accidentes', ascending=False).head(5)
    
    t = time.time() - start
    print("--- TOP 5 PUNTOS MÁS PELIGROSOS ENCONTRADOS (PANDAS) ---")
    print(top_5.to_string(index=False))
    print(f"⏱️ Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n🚀 [PYSPARK] Iniciando entorno de Spark para Consulta 4...")
    start_spark_init = time.time()
    
    # --- MEDICIÓN: Levantamiento de Spark ---
    spark = SparkSession.builder \
        .appName("C4") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    t_init = time.time() - start_spark_init
    print(f"⏱️ Tiempo de inicialización (Levantar Spark): {t_init:.4f} segundos")
    
    # --- MEDICIÓN: Proceso Real ---
    print(f"\n🔄 Procesando agregación espacial con PySpark...")
    start_process = time.time()
    
    # Optimización: Leemos solo las columnas requeridas explotando el formato Parquet
    df = spark.read.parquet(path).select("Start_Lat", "Start_Lng", "Severity")
    
    res = df.filter((F.col("Severity") >= 3) & (F.col("Start_Lat").isNotNull())) \
            .withColumn("Lat_Grid", F.round(F.col("Start_Lat"), 2)) \
            .withColumn("Lng_Grid", F.round(F.col("Start_Lng"), 2)) \
            .groupBy("Lat_Grid", "Lng_Grid") \
            .count() \
            .orderBy(F.col("count").desc()) \
            .limit(5) \
            .collect()
            
    t_process = time.time() - start_process
    t_total = t_init + t_process
    
    print(f"\n--- TOP 5 PUNTOS MÁS PELIGROSOS ENCONTRADOS (PYSPARK) ---")
    for r in res:
        print(f"Lat: {r['Lat_Grid']} | Lng: {r['Lng_Grid']} -> Accidentes: {r['count']}")
        
    print(f"⏱️ Tiempo Real de Proceso PySpark: {t_process:.4f} segundos")
    print(f"⏱️ Tiempo Total Acumulado PySpark: {t_total:.4f} segundos")
    
    spark.stop()
    return t_process, t_total

if __name__ == "__main__":
    # Obtener de forma segura el directorio base del proyecto 'mia_bigdata_project'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        BASE_DIR = os.path.dirname(script_dir)
    else:
        BASE_DIR = script_dir
        
    p = os.path.join(BASE_DIR, "data", "raw", "us_accidents.parquet")
    
    # Ejecuciones de Benchmark
    t_pandas = ejecutar_pandas(p)
    t_spark_proc, t_spark_total = ejecutar_pyspark(p)
    
    # Comparativas de velocidad (Speedup)
    print(f"\n📊 --- CONCLUSIONES DEL BENCHMARK CONSULTA 4 ---")
    print(f"📈 SPEEDUP REAL (Solo cómputo): {t_pandas / t_spark_proc:.2f}x más rápido con PySpark")
    print(f"📉 SPEEDUP GLOBAL (Incluyendo sobrecosto de levantar Spark): {t_pandas / t_spark_total:.2f}x")
