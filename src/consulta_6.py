"""
Ranking de autopistas críticas y tiempo de despeje: Normalizar las vías de la columna
Street (ej. I-95, I-5, US-101 ) para elaborar el Top 10 de autopistas con mayor índice de
siniestros graves (Severity ≥ 3) y reportar su tiempo promedio de despeje vial.
"""
"""
MIA - Big Data — Universidad de Palermo
GRUPO 4 - Consulta 6: Ranking de autopistas críticas (Top 10) y tiempo de despeje
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n🚀 [PANDAS] Ejecutando Consulta 6...")
    start_time = time.time()
    
    df = pd.read_parquet(path, columns=['Street', 'Severity', 'Start_Time', 'End_Time'])
    df_graves = df[(df['Severity'] >= 3) & (df['Street'].notna())].copy()
    
    # Conversión robusta de tipos temporales en Pandas
    df_graves['Start_Time'] = pd.to_datetime(df_graves['Start_Time'], errors='coerce')
    df_graves['End_Time'] = pd.to_datetime(df_graves['End_Time'], errors='coerce')
    df_graves['Despeje_Min'] = (df_graves['End_Time'] - df_graves['Start_Time']).dt.total_seconds() / 60.0
    
    top_10 = df_graves.groupby('Street').agg(
        Accidentes=('Severity', 'count'),
        Despeje_Promedio=('Despeje_Min', 'mean')
    ).reset_index().sort_values(by='Accidentes', ascending=False).head(10)
    
    t = time.time() - start_time
    print("--- TOP 10 AUTOPISTAS CRÍTICAS (PANDAS) ---")
    print(top_10.to_string(index=False))
    print(f"⏱️ Tiempo Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n🚀 [PYSPARK] Ejecutando Consulta 6...")
    start_time = time.time()
    
    # Optimizamos el motor configurando la política de tiempos a modo LEGACY
    spark = SparkSession.builder \
        .appName("C6") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
        
    df = spark.read.parquet(path)
    
    # Spark procesa el casteo de strings complejos a timestamp de manera nativa tolerando la política legacy
    res = df.filter((F.col("Severity") >= 3) & (F.col("Street").isNotNull())) \
            .withColumn("Start_TS", F.to_timestamp(F.col("Start_Time"))) \
            .withColumn("End_TS", F.to_timestamp(F.col("End_Time"))) \
            .withColumn("Despeje_Min", (F.unix_timestamp("End_TS") - F.unix_timestamp("Start_TS")) / 60.0) \
            .groupBy("Street") \
            .agg(F.count("Severity").alias("Accidentes"), F.mean("Despeje_Min").alias("Despeje_Promedio")) \
            .orderBy(F.col("Accidentes").desc()) \
            .limit(10) \
            .collect()
            
    t = time.time() - start_time
    print("\n--- TOP 10 AUTOPISTAS CRÍTICAS (PYSPARK) ---")
    for r in res:
        prom = r['Despeje_Promedio'] if r['Despeje_Promedio'] is not None else 0.0
        print(f"Calle: {r['Street']} | Accidentes: {r['Accidentes']} | Despeje Promedio: {prom:.2f} min")
        
    print(f"⏱️ Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n📈 SPEEDUP CONSULTA 6: {t_p / t_s:.2f}x")
