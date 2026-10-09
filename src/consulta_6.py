"""
MIA - Big Data - Universidad de Palermo
GRUPO 4 - Consulta 6: Ranking de autopistas criticas (Top 10) y tiempo de despeje
Consigna: Normalizar las vias de la columna Street (ej. I-95, I-5, US-101) para elaborar el Top 10
de autopistas con mayor indice de siniestros graves (Severity >= 3) y reportar su tiempo promedio de despeje vial.
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n[START] [PANDAS] Ejecutando Consulta 6 (Normalizacion Regex + Indice de Gravedad)...")
    start_time = time.time()
    
    # 1. Lectura inteligente de columnas necesarias
    df = pd.read_parquet(path, columns=['Street', 'Severity', 'Start_Time', 'End_Time'])
    df = df[df['Street'].notna()].copy()
    
    # 2. Normalizacion de autopistas: aislar prefijos de troncales (I-XX, US-XX)
    df['highway'] = df['Street'].str.extract(r'\b(I-\d+|US-\d+)\b', expand=False)
    df_hw = df[df['highway'].notna()].copy()
    
    # 3. Calculo de tiempo de despeje vial
    df_hw['Start_Time'] = pd.to_datetime(df_hw['Start_Time'], errors='coerce')
    df_hw['End_Time'] = pd.to_datetime(df_hw['End_Time'], errors='coerce')
    df_hw['Despeje_Min'] = (df_hw['End_Time'] - df_hw['Start_Time']).dt.total_seconds() / 60.0
    
    # Filtro de robustez temporal
    df_hw = df_hw[df_hw['Despeje_Min'].notna() & (df_hw['Despeje_Min'] > 0)].copy()
    df_hw['es_grave'] = df_hw['Severity'] >= 3
    
    # 4. Agrupacion por autopista normalizada
    res = df_hw.groupby('highway').agg(
        accidentes_totales=('Severity', 'count'),
        accidentes_graves=('es_grave', 'sum'),
        tiempo_despeje_promedio_min=('Despeje_Min', 'mean')
    ).reset_index()
    
    # Filtro de significancia estadistica (soporte minimo >= 100 siniestros)
    res = res[res['accidentes_totales'] >= 100].copy()
    res['indice_gravedad'] = res['accidentes_graves'] / res['accidentes_totales']
    
    # Ranking principal: Ordenado por Indice de Gravedad Relativo
    top_10 = res.sort_values(by='indice_gravedad', ascending=False).head(10)
    
    t = time.time() - start_time
    print("=====================================================================================")
    print("   TOP 10 AUTOPISTAS CRITICAS (ORDENADO POR INDICE DE GRAVEDAD - PANDAS)")
    print("=====================================================================================")
    print(top_10.to_string(index=False))
    print(f"[TIME] Tiempo Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n[START] [PYSPARK] Ejecutando Consulta 6 (Normalizacion Regex + Indice de Gravedad)...")
    start_time = time.time()
    
    spark = SparkSession.builder \
        .appName("C6_Grupo4") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
        
    df = spark.read.parquet(path)
    
    # 1. Normalizacion distribuida por Regex
    df_hw = df.withColumn("highway", F.regexp_extract(F.col("Street"), r"\b(I-\d+|US-\d+)\b", 1)) \
              .filter((F.col("highway").isNotNull()) & (F.col("highway") != ""))
              
    # 2. Calculo de tiempos de despeje y marca de gravedad
    df_hw = df_hw.withColumn("Start_TS", F.to_timestamp("Start_Time")) \
                 .withColumn("End_TS", F.to_timestamp("End_Time")) \
                 .withColumn("Despeje_Min", (F.unix_timestamp("End_TS") - F.unix_timestamp("Start_TS")) / 60.0) \
                 .filter(F.col("Despeje_Min").isNotNull() & (F.col("Despeje_Min") > 0)) \
                 .withColumn("es_grave", F.when(F.col("Severity") >= 3, 1).otherwise(0))
                 
    # 3. Agregacion y calculo de Indice de Gravedad
    res = df_hw.groupBy("highway").agg(
        F.count("*").alias("accidentes_totales"),
        F.sum("es_grave").alias("accidentes_graves"),
        F.mean("Despeje_Min").alias("tiempo_despeje_promedio_min")
    ).filter(F.col("accidentes_totales") >= 100) \
     .withColumn("indice_gravedad", F.col("accidentes_graves") / F.col("accidentes_totales")) \
     .orderBy(F.col("indice_gravedad").desc()) \
     .limit(10)
     
    resultados = res.collect()
    
    t = time.time() - start_time
    print("=====================================================================================")
    print("   TOP 10 AUTOPISTAS CRITICAS (ORDENADO POR INDICE DE GRAVEDAD - PYSPARK)")
    print("=====================================================================================")
    for r in resultados:
        print(f"Highway: {r['highway']:<6} | Totales: {r['accidentes_totales']:<5} | Graves: {r['accidentes_graves']:<5} | Despeje Prom: {r['tiempo_despeje_promedio_min']:.2f} min | Indice: {r['indice_gravedad']:.4f}")
        
    print(f"[TIME] Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n[SPEEDUP] SPEEDUP CONSULTA 6: {t_p / max(t_s, 0.0001):.2f}x")

