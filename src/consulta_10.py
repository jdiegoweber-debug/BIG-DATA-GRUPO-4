# =============================================================================
# CONSULTA 10
# =============================================================================
# CONSIGNA OFICIAL:
# Velocidad de despeje según iluminación y tipo de día: Comparar el tiempo medio de
# despeje (End_Time−Start_Time) en condiciones diurnas versus nocturnas (Sunrise_Sunset,
# Astronomical_Twilight), contrastando días hábiles con fines de semana.
# =============================================================================

import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n🚀 [PANDAS] Ejecutando Consulta 10...")
    start_time = time.time()
    
    # 1. Selección de columnas optimizada explotando el formato Parquet
    df = pd.read_parquet(path, columns=['Start_Time', 'End_Time', 'Sunrise_Sunset'])
    df = df[df['Sunrise_Sunset'].notna()].copy()
    
    # 2. Transformación robusta de datos temporales
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df['Despeje_Min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0
    
    # Mapeo de día en Pandas: 0=Lunes, 6=Domingo
    df['Tipo_Dia'] = df['Start_Time'].dt.dayofweek.apply(lambda x: 'Habil' if x < 5 else 'Fin de Semana')
    
    resultado = df.groupby(['Sunrise_Sunset', 'Tipo_Dia'])['Despeje_Min'].mean().reset_index()
    resultado.rename(columns={'Sunrise_Sunset': 'Iluminacion', 'Despeje_Min': 'Despeje_Promedio_Min'}, inplace=True)
    
    t = time.time() - start_time
    print("--- RESULTADOS VELOCIDAD DE DESPEJE (PANDAS) ---")
    print(resultado.to_string(index=False))
    print(f"⏱️ Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n🚀 [PYSPARK] Iniciando entorno de Spark para Consulta 10...")
    start_spark_init = time.time()
    
    # --- MEDICIÓN: Levantamiento de Spark con política LEGACY obligatoria ---
    spark = SparkSession.builder \
        .appName("C10") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
        
    t_init = time.time() - start_spark_init
    print(f"⏱️ Tiempo de inicialización (Levantar Spark): {t_init:.4f} segundos")
    
    # --- MEDICIÓN: Proceso Real ---
    print(f"\n🔄 Procesando métricas temporales de despeje con PySpark...")
    start_process = time.time()
    
    # Proyección temprana de columnas para eludir lecturas innecesarias en el disco mecánico
    df = spark.read.parquet(path).select("Start_Time", "End_Time", "Sunrise_Sunset")
    
    # Mapeo de día en Spark: 1=Domingo, 7=Sábado
    df_transformado = df.filter(F.col("Sunrise_Sunset").isNotNull()) \
        .withColumn("Start_TS", F.to_timestamp(F.col("Start_Time"))) \
        .withColumn("End_TS", F.to_timestamp(F.col("End_Time"))) \
        .withColumn("Despeje_Min", (F.unix_timestamp("End_TS") - F.unix_timestamp("Start_TS")) / 60.0) \
        .withColumn("Num_Dia", F.dayofweek(F.col("Start_TS"))) \
        .withColumn("Tipo_Dia", F.when((F.col("Num_Dia") == 1) | (F.col("Num_Dia") == 7), "Fin de Semana").otherwise("Habil"))
        
    df_res = df_transformado.groupBy("Sunrise_Sunset", "Tipo_Dia") \
        .agg(F.mean("Despeje_Min").alias("Despeje_Promedio_Min")) \
        .orderBy("Sunrise_Sunset", "Tipo_Dia")
        
    resultados = df_res.collect() # Acción terminal
    
    t_process = time.time() - start_process
    t_total = t_init + t_process
    
    print("\n--- RESULTADOS VELOCIDAD DE DESPEJE (PYSPARK) ---")
    for r in resultados:
        prom = r['Despeje_Promedio_Min'] if r['Despeje_Promedio_Min'] is not None else 0.0
        print(f"Iluminación: {r['Sunrise_Sunset']} | Tipo de Día: {r['Tipo_Dia']} | Despeje: {prom:.2f} min")
        
    print(f"⏱️ Tiempo Real de Proceso PySpark: {t_process:.4f} segundos")
    print(f"⏱️ Tiempo Total Acumulado PySpark: {t_total:.4f} segundos")
    
    spark.stop()
    return t_process, t_total

if __name__ == "__main__":
    # Obtener de forma segura la raíz dinámica de tu proyecto corporativo
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
    print(f"\n📊 --- CONCLUSIONES DEL BENCHMARK CONSULTA 10 ---")
    print(f"📈 SPEEDUP REAL (Solo cómputo): {t_pandas / t_spark_proc:.2f}x más rápido con PySpark")
    print(f"📉 SPEEDUP GLOBAL (Incluyendo sobrecosto de levantar Spark): {t_pandas / t_spark_total:.2f}x")
