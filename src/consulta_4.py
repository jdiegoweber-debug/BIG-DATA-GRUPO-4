
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

"""
MIA - Big Data — Universidad de Palermo
GRUPO 4 - Consulta 4: Hotspots espaciales (Top 5 puntos más peligrosos)
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n🚀 [PANDAS] Ejecutando Consulta 4...")
    start = time.time()
    
    df = pd.read_parquet(path, columns=['Start_Lat', 'Start_Lng', 'Severity'])
    df_graves = df[(df['Severity'] >= 3) & (df['Start_Lat'].notna())].copy()
    df_graves['Lat_Grid'] = df_graves['Start_Lat'].round(2)
    df_graves['Lng_Grid'] = df_graves['Start_Lng'].round(2)
    
    top_5 = df_graves.groupby(['Lat_Grid', 'Lng_Grid']).size().reset_index(name='Accidentes').sort_values(by='Accidentes', ascending=False).head(5)
    
    t = time.time() - start
    print(top_5.to_string(index=False))
    print(f"⏱️ Tiempo Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n🚀 [PYSPARK] Ejecutando Consulta 4...")
    start = time.time()
    
    spark = SparkSession.builder.appName("C4").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()
    df = spark.read.parquet(path)
    
    res = df.filter((F.col("Severity") >= 3) & (F.col("Start_Lat").isNotNull())) \
            .withColumn("Lat_Grid", F.round(F.col("Start_Lat"), 2)) \
            .withColumn("Lng_Grid", F.round(F.col("Start_Lng"), 2)) \
            .groupBy("Lat_Grid", "Lng_Grid").count().orderBy(F.col("count").desc()).limit(5).collect()
            
    t = time.time() - start
    for r in res:
        print(f"Lat: {r['Lat_Grid']} | Lng: {r['Lng_Grid']} -> Accidentes: {r['count']}")
        
    print(f"⏱️ Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n📈 SPEEDUP CONSULTA 4: {t_p / t_s:.2f}x")
