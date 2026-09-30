"""
Velocidad de despeje según iluminación y tipo de día: Comparar el tiempo medio de
despeje (End_Time−Start_Time) en condiciones diurnas versus nocturnas (Sunrise_Sunset,
Astronomical_Twilight), contrastando días hábiles con fines de semana.cls
""""""
MIA - Big Data — Universidad de Palermo
GRUPO 4 - Consulta 10: Velocidad de despeje según iluminación y tipo de día
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n🚀 [PANDAS] Ejecutando Consulta 10...")
    start_time = time.time()
    
    df = pd.read_parquet(path, columns=['Start_Time', 'End_Time', 'Sunrise_Sunset'])
    df = df[df['Sunrise_Sunset'].notna()].copy()
    
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df['Despeje_Min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0
    
    # 0=Lunes, 6=Domingo
    df['Tipo_Dia'] = df['Start_Time'].dt.dayofweek.apply(lambda x: 'Habil' if x < 5 else 'Fin de Semana')
    
    resultado = df.groupby(['Sunrise_Sunset', 'Tipo_Dia'])['Despeje_Min'].mean().reset_index()
    resultado.rename(columns={'Sunrise_Sunset': 'Iluminacion', 'Despeje_Min': 'Despeje_Promedio_Min'}, inplace=True)
    
    t = time.time() - start_time
    print("--- RESULTADOS VELOCIDAD DE DESPEJE (PANDAS) ---")
    print(resultado.to_string(index=False))
    print(f"⏱️ Tiempo Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n🚀 [PYSPARK] Ejecutando Consulta 10...")
    start_time = time.time()
    
    spark = SparkSession.builder \
        .appName("C10") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
        
    df = spark.read.parquet(path)
    
    # dayofweek en Spark: 1=Domingo, 7=Sábado
    df_transformado = df.filter(F.col("Sunrise_Sunset").isNotNull()) \
        .withColumn("Start_TS", F.to_timestamp(F.col("Start_Time"))) \
        .withColumn("End_TS", F.to_timestamp(F.col("End_Time"))) \
        .withColumn("Despeje_Min", (F.unix_timestamp("End_TS") - F.unix_timestamp("Start_TS")) / 60.0) \
        .withColumn("Num_Dia", F.dayofweek(F.col("Start_TS"))) \
        .withColumn("Tipo_Dia", F.when((F.col("Num_Dia") == 1) | (F.col("Num_Dia") == 7), "Fin de Semana").otherwise("Habil"))
        
    df_res = df_transformado.groupBy("Sunrise_Sunset", "Tipo_Dia") \
        .agg(F.mean("Despeje_Min").alias("Despeje_Promedio_Min")) \
        .orderBy("Sunrise_Sunset", "Tipo_Dia")
        
    resultados = df_res.collect()
    
    t = time.time() - start_time
    print("\n--- RESULTADOS VELOCIDAD DE DESPEJE (PYSPARK) ---")
    for r in resultados:
        prom = r['Despeje_Promedio_Min'] if r['Despeje_Promedio_Min'] is not None else 0.0
        print(f"Iluminación: {r['Sunrise_Sunset']} | Tipo de Día: {r['Tipo_Dia']} | Despeje: {prom:.2f} min")
        
    print(f"⏱️ Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n📈 SPEEDUP CONSULTA 10: {t_p / t_s:.2f}x")
