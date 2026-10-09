# =============================================================================
# CONSULTA 4 (Obligatoria para todos los grupos)
# =============================================================================
# CONSIGNA OFICIAL:
# Hotspots espaciales: top 5 puntos mas peligrosos.
# Identificar y reportar las 5 coordenadas geograficas mas peligrosas (a nivel nacional 
# o en un estado de alto flujo como California o Texas) dado un radio geografico de 
# influencia (ej. R = 2.0 km o 5.0 km). El equipo debe definir formalmente el criterio 
# de peligrosidad (densidad absoluta, concentracion de accidentes con Severity >= 3, 
# o indice ponderado) y justificar el metodo espacial implementado.
#
# JUSTIFICACION METODOLOGICA Y ESPACIAL DEL GRUPO 4 (DOBLE ENFOQUE):
# 1. ENFOQUE A (Severidad Critica Nacional - R ~ 2.0 km):
#    Filtra siniestros con Severity >= 3 para aislar riesgo de vida y colapso de autopistas
#    federales (I-95, I-105, I-75/I-85). Discretiza en celdas de 0.01 grados (~1.1-2.2 km).
# 2. ENFOQUE B (Densidad Absoluta Estatal CA - R ~ 5.0 km):
#    Mide el volumen bruto de accidentes totales en California agrupados en macro-celdas de
#    0.10 grados (~11.1 km de diametro) centradas en X.X50, aislando la maxima congestion.
# =============================================================================

"""
MIA - Big Data - Universidad de Palermo
GRUPO 4 - Consulta 4: Hotspots espaciales (Doble Enfoque Metodologico)
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n[START] [PANDAS] Ejecutando Consulta 4...")
    start = time.time()
    
    # 1. Carga optimizada
    df = pd.read_parquet(path, columns=['Start_Lat', 'Start_Lng', 'Severity', 'State'])
    
    # --- ENFOQUE A: Severidad Critica Nacional (Severity >= 3, R ~ 2.0 km) ---
    df_graves = df[(df['Severity'] >= 3) & (df['Start_Lat'].notna())].copy()
    df_graves['Lat_Grid'] = df_graves['Start_Lat'].round(2)
    df_graves['Lng_Grid'] = df_graves['Start_Lng'].round(2)
    top_5_a = df_graves.groupby(['Lat_Grid', 'Lng_Grid']).size().reset_index(name='Accidentes').sort_values(by='Accidentes', ascending=False).head(5)
    
    # --- ENFOQUE B: Densidad Absoluta Total en California (R ~ 5.0 km) ---
    df_ca = df[(df['State'] == 'CA') & (df['Start_Lat'].notna())].copy()
    df_ca['Lat_Bin'] = (df_ca['Start_Lat'] // 0.1) * 0.1 + 0.05
    df_ca['Lng_Bin'] = (df_ca['Start_Lng'] // 0.1) * 0.1 + 0.05
    top_5_b = df_ca.groupby(['Lat_Bin', 'Lng_Bin']).size().reset_index(name='Accidentes').sort_values(by='Accidentes', ascending=False).head(5)
    
    t = time.time() - start
    
    print("\n" + "="*70)
    print(" ENFOQUE A: TOP 5 PUNTOS MAS PELIGROSOS NACIONALES (SEVERITY >= 3, R ~ 2 km)")
    print("="*70)
    print(top_5_a.to_string(index=False))
    
    print("\n" + "="*70)
    print(" ENFOQUE B: TOP 5 HOTSPOTS POR DENSIDAD ABSOLUTA EN CALIFORNIA (R ~ 5 km)")
    print("="*70)
    print(top_5_b.to_string(index=False))
    
    print(f"\n[TIME] Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n[START] [PYSPARK] Ejecutando Consulta 4...")
    start = time.time()
    
    spark = SparkSession.builder.appName("C4").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()
    df = spark.read.parquet(path)
    
    # Enfoque A: Spark
    res_a = df.filter((F.col("Severity") >= 3) & (F.col("Start_Lat").isNotNull())) \
              .withColumn("Lat_Grid", F.round(F.col("Start_Lat"), 2)) \
              .withColumn("Lng_Grid", F.round(F.col("Start_Lng"), 2)) \
              .groupBy("Lat_Grid", "Lng_Grid").count().orderBy(F.col("count").desc()).limit(5).collect()
              
    # Enfoque B: Spark
    res_b = df.filter((F.col("State") == "CA") & (F.col("Start_Lat").isNotNull())) \
              .withColumn("Lat_Bin", F.round(F.floor(F.col("Start_Lat") / 0.1) * 0.1 + 0.05, 3)) \
              .withColumn("Lng_Bin", F.round(F.floor(F.col("Start_Lng") / 0.1) * 0.1 + 0.05, 3)) \
              .groupBy("Lat_Bin", "Lng_Bin").count().orderBy(F.col("count").desc()).limit(5).collect()
            
    t = time.time() - start
    print("\n--- PYSPARK ENFOQUE A (SEVERIDAD CRITICA NACIONAL) ---")
    for r in res_a:
        print(f"Lat: {r['Lat_Grid']} | Lng: {r['Lng_Grid']} -> Accidentes Graves: {r['count']}")
        
    print("\n--- PYSPARK ENFOQUE B (DENSIDAD ABSOLUTA EN CALIFORNIA) ---")
    for r in res_b:
        print(f"Lat: {r['Lat_Bin']} | Lng: {r['Lng_Bin']} -> Accidentes Totales: {r['count']}")
        
    print(f"\n[TIME] Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n[SPEEDUP] SPEEDUP CONSULTA 4: {t_p / t_s:.2f}x")
