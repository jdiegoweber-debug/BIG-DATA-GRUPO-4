"""
MIA - Big Data - Universidad de Palermo
GRUPO 4 - Consulta 3: Busqueda espacial por radio dado un punto mediante la distancia de Haversine
"""
import os
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Punto geografico de prueba: Los Angeles, California
LAT0 = 34.05
LNG0 = -118.24
RADIO_R = 25.0  # Kilometros
RADIO_TIERRA = 6371.0  # Radio medio de la Tierra en km

def ejecutar_pandas(path):
    print(f"\n[START] [PANDAS] Ejecutando Consulta 3 (Haversine R={RADIO_R} km)...")
    start_time = time.time()
    
    # 1. Lectura inteligente de columnas necesarias
    df = pd.read_parquet(path, columns=['Start_Lat', 'Start_Lng', 'Severity'])
    df = df[df['Start_Lat'].notna() & df['Start_Lng'].notna()].copy()
    
    # 2. Convertir coordenadas de grados a radianes para la formula
    lat1 = np.radians(df['Start_Lat'])
    lng1 = np.radians(df['Start_Lng'])
    lat2 = np.radians(LAT0)
    lng2 = np.radians(LNG0)
    
    # 3. Aplicar formula matematica de Haversine vectorizada en Pandas
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['Distancia_KM'] = RADIO_TIERRA * c
    
    # 4. Filtrar por radio y ordenar por proximidad
    resultados = df[df['Distancia_KM'] <= RADIO_R]
    top_10 = resultados.sort_values(by='Distancia_KM').head(10)
    
    t = time.time() - start_time
    print(f"--- PRIMEROS 10 ACCIDENTES MAS CERCANOS ENCONTRADOS (PANDAS) ---")
    print(top_10.to_string(index=False))
    print(f"Total accidentes en el radio: {len(resultados)}")
    print(f"[TIME] Tiempo Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print(f"\n[START] [PYSPARK] Ejecutando Consulta 3 (Haversine R={RADIO_R} km)...")
    start_time = time.time()
    
    spark = SparkSession.builder.appName("C3").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()
    df = spark.read.parquet(path)
    
    # 1. Filtrar nulos iniciales
    df_valid = df.filter(F.col("Start_Lat").isNotNull() & F.col("Start_Lng").isNotNull())
    
    # 2. Aplicar Haversine nativo mediante funciones matematicas distribuidas de Spark
    # Convertimos grados a radianes usando F.radians()
    df_dist = df_valid.withColumn("dlat", F.radians(F.lit(LAT0)) - F.radians(F.col("Start_Lat"))) \
                      .withColumn("dlng", F.radians(F.lit(LNG0)) - F.radians(F.col("Start_Lng"))) \
                      .withColumn("a", F.sin(F.col("dlat") / 2)**2 + 
                                       F.cos(F.radians(F.col("Start_Lat"))) * F.cos(F.radians(F.lit(LAT0))) * F.sin(F.col("dlng") / 2)**2) \
                      .withColumn("c", 2 * F.asin(F.sqrt(F.col("a")))) \
                      .withColumn("Distancia_KM", F.lit(RADIO_TIERRA) * F.col("c"))
    
    # 3. Filtrar por radio, seleccionar columnas, ordenar y aplicar accion terminal
    df_final = df_dist.filter(F.col("Distancia_KM") <= RADIO_R) \
                      .select("Start_Lat", "Start_Lng", "Severity", "Distancia_KM") \
                      .orderBy("Distancia_KM")
                      
    resultados = df_final.limit(10).collect()
    total_count = df_final.count()  # Forzamos conteo global para el benchmark
    
    t = time.time() - start_time
    print(f"\n--- PRIMEROS 10 ACCIDENTES MAS CERCANOS ENCONTRADOS (PYSPARK) ---")
    for r in resultados:
        print(f"Lat: {r['Start_Lat']} | Lng: {r['Start_Lng']} | Severidad: {r['Severity']} | Distancia: {r['Distancia_KM']:.2f} km")
        
    print(f"Total accidentes en el radio: {total_count}")
    print(f"[TIME] Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n[SPEEDUP] SPEEDUP CONSULTA 3: {t_p / t_s:.2f}x")
