# =============================================================================
# CONSULTA 6
# =============================================================================
# CONSIGNA OFICIAL:
# Ranking de autopistas críticas y tiempo de despeje: Normalizar las vías de la columna
# Street (ej. I-95, I-5, US-101 ) para elaborar el Top 10 de autopistas con mayor índice de
# siniestros graves (Severity >= 3) y reportar su tiempo promedio de despeje vial.
# =============================================================================

import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Patrón REGEX para normalizar autopistas interestatales (I-X) y rutas federales (US-X)
# Captura patrones como "I- 95", "I-95 N", "US 101" y los unifica en "I-95" o "US-101"
REGEX_AUTOPISAS = r'(I\s*-\s*\d+|US\s*-\s*\d+|US\s+\d+|I\s+\d+)'

def normalizar_street_pandas(series):
    # Reemplaza espacios extra y unifica formatos comunes en mayúsculas
    cleaned = series.str.upper().str.extract(REGEX_AUTOPISAS, expand=False)
    cleaned = cleaned.str.replace(r'\s+', '', regex=True) # Quita espacios intermedios (ej: I 95 -> I95)
    cleaned = cleaned.str.replace(r'([A-Z])(?=\d)', r'\1-', regex=True) # Agrega guión si falta (ej: I95 -> I-95)
    return cleaned.fillna(series) # Si no es autopista principal, conserva el nombre original

def ejecutar_pandas(path):
    print("\n🚀 [PANDAS] Ejecutando Consulta 6...")
    start_time = time.time()
    
    # 1. Lectura inteligente explotando proyecciones de Parquet
    df = pd.read_parquet(path, columns=['Street', 'Severity', 'Start_Time', 'End_Time'])
    df_graves = df[(df['Severity'] >= 3) & (df['Street'].notna())].copy()
    
    # 2. Normalización de vías requerida por la consigna
    df_graves['Street_Norm'] = normalizar_street_pandas(df_graves['Street'])
    
    # 3. Conversión robusta de tipos temporales
    df_graves['Start_Time'] = pd.to_datetime(df_graves['Start_Time'], errors='coerce')
    df_graves['End_Time'] = pd.to_datetime(df_graves['End_Time'], errors='coerce')
    df_graves['Despeje_Min'] = (df_graves['End_Time'] - df_graves['Start_Time']).dt.total_seconds() / 60.0
    
    top_10 = df_graves.groupby('Street_Norm').agg(
        Accidentes=('Severity', 'count'),
        Despeje_Promedio=('Despeje_Min', 'mean')
    ).reset_index().sort_values(by='Accidentes', ascending=False).head(10)
    
    t = time.time() - start_time
    print("--- TOP 10 AUTOPISTAS CRÍTICAS NORMALIZADAS (PANDAS) ---")
    print(top_10.to_string(index=False))
    print(f"⏱️ Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n🚀 [PYSPARK] Iniciando entorno de Spark para Consulta 6...")
    start_spark_init = time.time()
    
    # --- MEDICIÓN: Levantamiento de Spark con política LEGACY para parseo seguro ---
    spark = SparkSession.builder \
        .appName("C6") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
        
    t_init = time.time() - start_spark_init
    print(f"⏱️ Tiempo de inicialización (Levantar Spark): {t_init:.4f} segundos")
    
    # --- MEDICIÓN: Proceso Real ---
    print(f"\n🔄 Procesando y normalizando texto distribuido con PySpark...")
    start_process = time.time()
    
    # Selección temprana de columnas clave
    df = spark.read.parquet(path).select("Street", "Severity", "Start_Time", "End_Time")
    
    # Expresión regular equivalente en Spark para normalizar e incorporar el guión estándar
    df_norm = df.filter((F.col("Severity") >= 3) & (F.col("Street").isNotNull())) \
                .withColumn("Street_Upper", F.upper(F.col("Street"))) \
                .withColumn("Extracted", F.regexp_extract("Street_Upper", REGEX_AUTOPISAS, 1)) \
                .withColumn("Cleaned", F.regexp_replace("Extracted", r"\s+", "")) \
                .withColumn("Street_Norm", F.when(F.col("Cleaned") != "", 
                                            F.regexp_replace("Cleaned", r"([A-Z])(?=\d)", r"$1-"))
                                            .otherwise(F.col("Street")))
    
    # Tratamiento temporal y agregación analítica
    res = df_norm.withColumn("Start_TS", F.to_timestamp(F.col("Start_Time"))) \
                 .withColumn("End_TS", F.to_timestamp(F.col("End_Time"))) \
                 .withColumn("Despeje_Min", (F.unix_timestamp("End_TS") - F.unix_timestamp("Start_TS")) / 60.0) \
                 .groupBy("Street_Norm") \
                 .agg(F.count("Severity").alias("Accidentes"), F.mean("Despeje_Min").alias("Despeje_Promedio")) \
                 .orderBy(F.col("Accidentes").desc()) \
                 .limit(10) \
                 .collect()
            
    t_process = time.time() - start_process
    t_total = t_init + t_process
    
    print("\n--- TOP 10 AUTOPISTAS CRÍTICAS NORMALIZADAS (PYSPARK) ---")
    for r in res:
        prom = r['Despeje_Promedio'] if r['Despeje_Promedio'] is not None else 0.0
        print(f"Calle: {r['Street_Norm']} | Accidentes: {r['Accidentes']} | Despeje Promedio: {prom:.2f} min")
        
    print(f"⏱️ Tiempo Real de Proceso PySpark: {t_process:.4f} segundos")
    print(f"⏱️ Tiempo Total Acumulado PySpark: {t_total:.4f} segundos")
    
    spark.stop()
    return t_process, t_total

if __name__ == "__main__":
    # Obtener raíz dinámica del proyecto de forma segura
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
    print(f"\n📊 --- CONCLUSIONES DEL BENCHMARK CONSULTA 6 ---")
    print(f"📈 SPEEDUP REAL (Solo cómputo): {t_pandas / t_spark_proc:.2f}x más rápido con PySpark")
    print(f"📉 SPEEDUP GLOBAL (Incluyendo sobrecosto de levantar Spark): {t_pandas / t_spark_total:.2f}x")
