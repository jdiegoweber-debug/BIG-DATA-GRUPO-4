"""
MIA - Big Data — Universidad de Palermo
GRUPO 4 - Consulta 3: Búsqueda espacial por radio dado un punto mediante la distancia de Haversine
"""
import os
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Punto geográfico de prueba: Los Ángeles, California
LAT0 = 34.05
LNG0 = -118.24
RADIO_R = 25.0  # Kilómetros
RADIO_TIERRA = 6371.0  # Radio medio de la Tierra en km

def ejecutar_pandas(path):
    print(f"\n🚀 [PANDAS] Ejecutando Consulta 3 (Haversine R={RADIO_R} km)...")
    start_time = time.time()
    
    # 1. Lectura inteligente de columnas necesarias
    df = pd.read_parquet(path, columns=['Start_Lat', 'Start_Lng', 'Severity'])
    df = df[df['Start_Lat'].notna() & df['Start_Lng'].notna()].copy()
    
    # 2. Convertir coordenadas de grados a radianes para la fórmula
    lat1 = np.radians(df['Start_Lat'])
    lng1 = np.radians(df['Start_Lng'])
    lat2 = np.radians(LAT0)
    lng2 = np.radians(LNG0)
    
    # 3. Aplicar fórmula matemática de Haversine vectorizada en Pandas
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['Distancia_KM'] = RADIO_TIERRA * c
    
    # 4. Filtrar por radio y ordenar por proximidad
    resultados = df[df['Distancia_KM'] <= RADIO_R]
    top_10 = resultados.sort_values(by='Distancia_KM').head(10)
    
    t = time.time() - start_time
    print(f"--- PRIMEROS 10 ACCIDENTES MÁS CERCANOS ENCONTRADOS (PANDAS) ---")
    print(top_10.to_string(index=False))
    print(f"Total accidentes en el radio: {len(resultados)}")
    print(f"⏱️ Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print(f"\n🚀 [PYSPARK] Iniciando entorno de Spark...")
    start_spark_init = time.time()
    
    # --- MEDICIÓN: Levantamiento de Spark ---
    spark = SparkSession.builder \
        .appName("C3") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    t_init = time.time() - start_spark_init
    print(f"⏱️ Tiempo de inicialización (Levantar Spark): {t_init:.4f} segundos")
    
    # --- MEDICIÓN: Proceso Real ---
    print(f"\n🔄 Procesando datos distribuidos con PySpark...")
    start_process = time.time()
    
    # Optimización: seleccionamos solo columnas necesarias inmediatamente
    df = spark.read.parquet(path).select("Start_Lat", "Start_Lng", "Severity")
    
    # 1. Filtrar nulos iniciales
    df_valid = df.filter(F.col("Start_Lat").isNotNull() & F.col("Start_Lng").isNotNull())
    
    # 2. Aplicar Haversine nativo mediante funciones matemáticas distribuidas de Spark
    df_dist = df_valid.withColumn("dlat", F.radians(F.lit(LAT0)) - F.radians(F.col("Start_Lat"))) \
                      .withColumn("dlng", F.radians(F.lit(LNG0)) - F.radians(F.col("Start_Lng"))) \
                      .withColumn("a", F.sin(F.col("dlat") / 2)**2 + 
                                       F.cos(F.radians(F.col("Start_Lat"))) * F.cos(F.radians(F.lit(LAT0))) * F.sin(F.col("dlng") / 2)**2) \
                      .withColumn("c", 2 * F.asin(F.sqrt(F.col("a")))) \
                      .withColumn("Distancia_KM", F.lit(RADIO_TIERRA) * F.col("c"))
    
    # 3. Filtrar por radio, ordenar y aplicar acción terminal
    df_final = df_dist.filter(F.col("Distancia_KM") <= RADIO_R) \
                      .select("Start_Lat", "Start_Lng", "Severity", "Distancia_KM") \
                      .orderBy("Distancia_KM")
                      
    resultados = df_final.limit(10).collect()
    total_count = df_final.count()  # Forzamos conteo global para el benchmark
    
    t_process = time.time() - start_process
    t_total = t_init + t_process
    
    print(f"\n--- PRIMEROS 10 ACCIDENTES MÁS CERCANOS ENCONTRADOS (PYSPARK) ---")
    for r in resultados:
        print(f"Lat: {r['Start_Lat']} | Lng: {r['Start_Lng']} | Severidad: {r['Severity']} | Distancia: {r['Distancia_KM']:.2f} km")
        
    print(f"Total accidentes en el radio: {total_count}")
    print(f"⏱️ Tiempo Real de Proceso PySpark: {t_process:.4f} segundos")
    print(f"⏱️ Tiempo Total Acumulado PySpark: {t_total:.4f} segundos")
    
    spark.stop()
    return t_process, t_total

if __name__ == "__main__":
    # Obtener de forma segura el directorio base del proyecto 'mia_bigdata_project'
    # Si se ejecuta desde 'src', sube un nivel; si se ejecuta desde la raíz, se queda ahí
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        BASE_DIR = os.path.dirname(script_dir)
    else:
        BASE_DIR = script_dir
        
    p = os.path.join(BASE_DIR, "data", "raw", "us_accidents.parquet")
    
    # Ejecuciones
    t_pandas = ejecutar_pandas(p)
    t_spark_proc, t_spark_total = ejecutar_pyspark(p)
    
    # Comparativas de velocidad (Speedup)
    print(f"\n📊 --- CONCLUSIONES DEL BENCHMARK ---")
    print(f"📈 SPEEDUP REAL (Solo cómputo): {t_pandas / t_spark_proc:.2f}x más rápido con PySpark")
    print(f"📉 SPEEDUP GLOBAL (Incluyendo sobrecosto de levantar Spark): {t_pandas / t_spark_total:.2f}x")
