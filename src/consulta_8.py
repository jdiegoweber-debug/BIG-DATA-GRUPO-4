# =============================================================================
# CONSULTA 8 (Grupo 4)
# =============================================================================
# CONSIGNA OFICIAL:
# Deteccion de dias con anomalias meteorologicas severas:
# Detectar dias y estados donde los siniestros superaron en mas de 3 desviaciones 
# estandar (>3 sigma) la media historica del estado, cruzando con variables climaticas 
# (Precipitation, Wind_Speed, Visibility) para corroborar eventos extremos.
#
# JUSTIFICACION METODOLOGICA DEL GRUPO 4 (DOBLE PERSPECTIVA ANALITICA):
# 1. PERSPECTIVA 1 — IMPACTO VOLUMETRICO ABSOLUTO (Ordenado por daily_accidents):
#    Identifica los dias de colapso neto de la infraestructura asistencial y autopistas.
#    Liderado por California (CA), con picos de hasta 2,823 siniestros en 24 horas (4.05x la media).
# 2. PERSPECTIVA 2 — RAREZA METEOROLOGICA EXTREMA (Ordenado por Z-Score):
#    Mide la magnitud estadistica pura de la perturbacion climatica (desvios sobre la media).
#    Liderado por tormentas polares e invernales en Kansas (Z = 20.57 sigma), Iowa (Z = 18.25 sigma)
#    y Ohio (Z = 15.62 sigma con vientos de 26.2 mph y visibilidad reducida a 1.29 millas).
# =============================================================================

"""
MIA - Big Data - Universidad de Palermo
GRUPO 4 - Consulta 8: Deteccion de dias con anomalias meteorologicas severas (> 3 sigma)
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def ejecutar_pandas(path):
    print("\n[START] [PANDAS] Ejecutando Consulta 8 (Anomalias > 3 Sigma con Doble Perspectiva)...")
    start_time = time.time()
    
    # 1. Carga optimizada con variables meteorologicas de la consigna
    cols = ['Start_Time', 'State', 'Precipitation(in)', 'Wind_Speed(mph)', 'Visibility(mi)']
    df = pd.read_parquet(path, columns=cols)
    df = df[df['Start_Time'].notna() & df['State'].notna()].copy()
    
    # Truncar Start_Time a fecha (YYYY-MM-DD)
    df['date'] = df['Start_Time'].astype(str).str.slice(0, 10)
    
    # 2. Agrupar por Estado y Fecha calculando metricas climaticas
    diario = df.groupby(['State', 'date']).agg(
        daily_accidents=('Start_Time', 'count'),
        mean_precipitation=('Precipitation(in)', 'mean'),
        mean_wind_speed=('Wind_Speed(mph)', 'mean'),
        mean_visibility=('Visibility(mi)', 'mean')
    ).reset_index()
    
    # 3. Estadisticas historicas del Estado (media y desvio estandar)
    stats = diario.groupby('State').agg(
        mean_historical=('daily_accidents', 'mean'),
        std_historical=('daily_accidents', 'std')
    ).reset_index()
    
    # 4. Cruzar datos y computar Z-Score formal
    merged = pd.merge(diario, stats, on='State')
    merged['z_score'] = (merged['daily_accidents'] - merged['mean_historical']) / merged['std_historical']
    
    # Filtrar anomalias severas (> 3 sigma)
    anomalias = merged[merged['z_score'] > 3.0].copy()
    
    # Perspectiva 1: Impacto Volumetrico Absoluto
    top_5_vol = anomalias.sort_values(by='daily_accidents', ascending=False).head(5)
    
    # Perspectiva 2: Rareza Estadistica Extrema (Z-Score)
    top_5_z = anomalias.sort_values(by='z_score', ascending=False).head(5)
    
    t = time.time() - start_time
    
    print("\n" + "="*85)
    print(" PERSPECTIVA 1: TOP 5 DIAS POR IMPACTO VOLUMETRICO ABSOLUTO (ACCIDENTES TOTALES)")
    print("="*85)
    cols_vol = ['State', 'date', 'daily_accidents', 'mean_historical', 'mean_precipitation', 'z_score']
    print(top_5_vol[cols_vol].to_string(index=False))
    
    print("\n" + "="*85)
    print(" PERSPECTIVA 2: TOP 5 DIAS POR RAREZA ESTADISTICA METEOROLOGICA (Z-SCORE > 3 SIGMA)")
    print("="*85)
    cols_z = ['State', 'date', 'daily_accidents', 'z_score', 'mean_precipitation', 'mean_wind_speed', 'mean_visibility']
    print(top_5_z[cols_z].to_string(index=False))
    
    print(f"\nTotal dias anomalos (> 3 sigma) detectados en toda la historia: {len(anomalias)}")
    print(f"[TIME] Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n[START] [PYSPARK] Ejecutando Consulta 8 (Funciones de Ventana Distribuidas)...")
    start_time = time.time()
    
    spark = SparkSession.builder.appName("C8").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()
    df = spark.read.parquet(path)
    
    # 1. Agrupacion diaria y climatica
    df_diario = df.filter(F.col("Start_Time").isNotNull() & F.col("State").isNotNull()) \
                  .withColumn("date", F.substring(F.col("Start_Time"), 1, 10)) \
                  .groupBy("State", "date") \
                  .agg(
                      F.count("Start_Time").alias("daily_accidents"),
                      F.mean("Precipitation(in)").alias("mean_precipitation"),
                      F.mean("Wind_Speed(mph)").alias("mean_wind_speed"),
                      F.mean("Visibility(mi)").alias("mean_visibility")
                  )
                  
    # 2. Window Functions por Estado
    win_state = Window.partitionBy("State")
    df_stats = df_diario.withColumn("mean_historical", F.mean("daily_accidents").over(win_state)) \
                        .withColumn("std_historical", F.stddev("daily_accidents").over(win_state))
                        
    # 3. Z-Score y filtrado > 3 sigma
    df_anomalias = df_stats.withColumn(
        "z_score", 
        (F.col("daily_accidents") - F.col("mean_historical")) / F.col("std_historical")
    ).filter(F.col("z_score") > 3.0)
    
    # Perspectiva 1: Spark
    res_vol = df_anomalias.orderBy(F.col("daily_accidents").desc()).limit(5).collect()
    
    # Perspectiva 2: Spark
    res_z = df_anomalias.orderBy(F.col("z_score").desc()).limit(5).collect()
    
    total_anomalias = df_anomalias.count()
    t = time.time() - start_time
    
    print("\n--- PYSPARK PERSPECTIVA 1 (IMPACTO VOLUMETRICO ABSOLUTO) ---")
    for r in res_vol:
        print(f"Estado: {r['State']} | Fecha: {r['date']} | Siniestros: {r['daily_accidents']} | Media Hist: {r['mean_historical']:.1f} | Z-Score: {r['z_score']:.2f} sigma")
        
    print("\n--- PYSPARK PERSPECTIVA 2 (RAREZA ESTADISTICA METEOROLOGICA) ---")
    for r in res_z:
        print(f"Estado: {r['State']} | Fecha: {r['date']} | Z-Score: {r['z_score']:.2f} sigma | Siniestros: {r['daily_accidents']} | Viento: {r['mean_wind_speed']:.1f} mph | Visibilidad: {r['mean_visibility']:.2f} mi")
        
    print(f"\nTotal dias anomalos en PySpark: {total_anomalias}")
    print(f"[TIME] Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n[SPEEDUP] SPEEDUP CONSULTA 8: {t_p / t_s:.2f}x")
