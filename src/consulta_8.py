# =============================================================================
# CONSULTA 8
# =============================================================================
# CONSIGNA OFICIAL:
# Detección de días con anomalías meteorológicas severas: Detectar días y estados
# donde los siniestros superaron en más de 3 desviaciones estándar (> 3σ) la media histórica
# del estado, cruzando con variables climáticas (Precipitation, Wind_Speed, Visibility)
# para corroborar eventos extremos.
# =============================================================================

import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def ejecutar_pandas(path):
    print("\n🚀 [PANDAS] Ejecutando Consulta 8 (Anomalías > 3 Sigma)...")
    start_time = time.time()
    
    # 1. Cargar datos necesarios optimizando la proyección de Parquet
    df = pd.read_parquet(path, columns=['Start_Time', 'State', 'Precipitation(in)', 'Visibility(mi)'])
    df = df[df['Start_Time'].notna() & df['State'].notna()].copy()
    
    # Extraemos solo la fecha (YYYY-MM-DD) mediante rebanado de string
    df['Fecha'] = df['Start_Time'].str.slice(0, 10)
    
    # 2. Agrupar por Día y Estado para contar siniestros y sacar promedios meteorológicos
    diario = df.groupby(['State', 'Fecha']).agg(
        Accidentes_Dia=('Start_Time', 'count'),
        Precipitacion_Media=('Precipitation(in)', 'mean'),
        Visibilidad_Media=('Visibility(mi)', 'mean')
    ).reset_index()
    
    # 3. Calcular la media y desviación estándar de accidentes históricos de cada Estado
    stats = diario.groupby('State').agg(
        Media_Historica=('Accidentes_Dia', 'mean'),
        Desvio_Historico=('Accidentes_Dia', 'std')
    ).reset_index()
    
    # 4. Cruzar los datos y aplicar el filtro de la anomalía de más de 3 Sigmas
    merged = pd.merge(diario, stats, on='State')
    merged['Umbral_Anomalia'] = merged['Media_Historica'] + (3 * merged['Desvio_Historico'])
    anomalias = merged[merged['Accidentes_Dia'] > merged['Umbral_Anomalia']]
    
    # Top 5 de días más anómalos a nivel nacional
    top_5 = anomalias.sort_values(by='Accidentes_Dia', ascending=False).head(5)
    
    t = time.time() - start_time
    print("--- TOP 5 DÍAS MÁS ANÓMALOS DETECTADOS (PANDAS) ---")
    print(top_5[['State', 'Fecha', 'Accidentes_Dia', 'Media_Historica', 'Precipitacion_Media']].to_string(index=False))
    print(f"Total días anómalos en toda la historia: {len(anomalias)}")
    print(f"⏱️ Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n🚀 [PYSPARK] Iniciando entorno de Spark para Consulta 8...")
    start_spark_init = time.time()
    
    # --- MEDICIÓN: Levantamiento de Spark ---
    spark = SparkSession.builder \
        .appName("C8") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    t_init = time.time() - start_spark_init
    print(f"⏱️ Tiempo de inicialización (Levantar Spark): {t_init:.4f} segundos")
    
    # --- MEDICIÓN: Proceso Real ---
    print(f"\n🔄 Procesando analítica de ventanas distribuida con PySpark...")
    start_process = time.time()
    
    # Selección temprana de columnas explotando el formato Parquet indexado
    df = spark.read.parquet(path).select("Start_Time", "State", "Precipitation(in)", "Visibility(mi)")
    
    # 1. Truncar fecha a nivel de día y agrupar por Estado/Fecha
    df_diario = df.filter(F.col("Start_Time").isNotNull() & F.col("State").isNotNull()) \
                  .withColumn("Fecha", F.substring(F.col("Start_Time"), 1, 10)) \
                  .groupBy("State", "Fecha") \
                  .agg(
                      F.count("Start_Time").alias("Accidentes_Dia"),
                      F.mean("Precipitation(in)").alias("Precipitacion_Media"),
                      F.mean("Visibility(mi)").alias("Visibilidad_Media")
                  )
                  
    # 2. Usar Funciones de Ventana (Window Functions) para calcular estadísticas históricas por Estado
    ventana_estado = Window.partitionBy("State")
    
    df_stats = df_diario.withColumn("Media_Historica", F.mean("Accidentes_Dia").over(ventana_estado)) \
                        .withColumn("Desvio_Historico", F.stddev("Accidentes_Dia").over(ventana_estado))
                        
    # 3. Filtrar registros que superen el umbral crítico (> 3 sigma)
    df_anomalias = df_stats.filter(F.col("Accidentes_Dia") > (F.col("Media_Historica") + (3 * F.col("Desvio_Historico"))))
    
    resultados = df_anomalias.orderBy(F.col("Accidentes_Dia").desc()).limit(5).collect()
    total_anomalias = df_anomalias.count() # Acción terminal para materializar el grafo completo
    
    t_process = time.time() - start_process
    t_total = t_init + t_process
    
    print("\n--- TOP 5 DÍAS MÁS ANÓMALOS DETECTADOS (PYSPARK) ---")
    for r in resultados:
        print(f"Estado: {r['State']} | Fecha: {r['Fecha']} | Accidentes: {r['Accidentes_Dia']} | Media Hist: {r['Media_Historica']:.1f} | Precipitación Prom: {r['Precipitacion_Media']:.2f} in")
        
    print(f"Total días anómalos en toda la historia: {total_anomalias}")
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
    print(f"\n📊 --- CONCLUSIONES DEL BENCHMARK CONSULTA 8 ---")
    print(f"📈 SPEEDUP REAL (Solo cómputo): {t_pandas / t_spark_proc:.2f}x más rápido con PySpark")
    print(f"📉 SPEEDUP GLOBAL (Incluyendo sobrecosto de levantar Spark): {t_pandas / t_spark_total:.2f}x")
