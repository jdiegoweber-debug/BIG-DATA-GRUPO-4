# =============================================================================
# CONSULTA 10 (Grupo 4)
# =============================================================================
# CONSIGNA OFICIAL:
# Velocidad de despeje segun iluminacion y tipo de dia:
# Comparar el tiempo medio de despeje (End_Time - Start_Time) en condiciones 
# diurnas versus nocturnas (Sunrise_Sunset, Astronomical_Twilight), contrastando 
# dias habiles con fines de semana.
#
# JUSTIFICACION METODOLOGICA DEL GRUPO 4:
# 1. MATRIZ EJECUTIVA (2x2 = 4 cuadrantes):
#    Mide el efecto macro de luz diurna (Sunrise_Sunset) vs dia laborable/fin de semana.
#    Demuestra que un siniestro nocturno en fin de semana promedia 720.07 min (~12 h),
#    duplicando con creces los 336.46 min de un dia habil diurno.
# 2. MATRIZ FOTOMETRICA DETALLADA (2x2x2 = 8 combinaciones):
#    Cruza Sunrise_Sunset con Astronomical_Twilight (crepusculo astronomico, -18° solar).
#    Revela que en oscuridad profunda total (NIGHT / NIGHT) durante fines de semana,
#    el despeje promedio asciende a 772.75 minutos (~12.88 horas) para 291,307 accidentes,
#    mientras que la presencia de luz crepuscular (NIGHT / DAY) amortigua el retraso a 345.65 min.
# =============================================================================

"""
MIA - Big Data - Universidad de Palermo
GRUPO 4 - Consulta 10: Velocidad de despeje segun iluminacion y tipo de dia
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def ejecutar_pandas(path):
    print("\n[START] [PANDAS] Ejecutando Consulta 10 (Analisis Fotometrico y Temporal)...")
    start_time = time.time()
    
    # 1. Carga optimizada
    cols = ['Start_Time', 'End_Time', 'Sunrise_Sunset', 'Astronomical_Twilight']
    df = pd.read_parquet(path, columns=cols)
    df = df[df['Sunrise_Sunset'].notna() & df['Astronomical_Twilight'].notna()].copy()
    
    # 2. Parseo de marcas temporales y calculo de duracion
    df['Start_DT'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_DT'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df = df[df['Start_DT'].notna() & df['End_DT'].notna()].copy()
    
    df['Despeje_Min'] = (df['End_DT'] - df['Start_DT']).dt.total_seconds() / 60.0
    # Filtrar anomalias de duracion negativa o espuria
    df = df[df['Despeje_Min'] >= 0].copy()
    
    # 3. Clasificacion temporal (0=Lunes, 6=Domingo)
    df['Tipo_Dia'] = df['Start_DT'].dt.dayofweek.apply(lambda x: 'DIA HABIL' if x < 5 else 'FIN DE SEMANA')
    
    # --- MATRIZ EJECUTIVA (4 CUADRANTES) ---
    res_ejecutiva = df.groupby(['Sunrise_Sunset', 'Tipo_Dia']).agg(
        accidentes_totales=('Start_DT', 'count'),
        despeje_promedio_min=('Despeje_Min', 'mean')
    ).reset_index()
    res_ejecutiva['despeje_promedio_horas'] = res_ejecutiva['despeje_promedio_min'] / 60.0
    
    # --- MATRIZ FOTOMETRICA COMPLETA (8 COMBINACIONES) ---
    res_fotometrica = df.groupby(['Sunrise_Sunset', 'Astronomical_Twilight', 'Tipo_Dia']).agg(
        accidentes_totales=('Start_DT', 'count'),
        despeje_promedio_min=('Despeje_Min', 'mean')
    ).reset_index()
    res_fotometrica['despeje_promedio_horas'] = res_fotometrica['despeje_promedio_min'] / 60.0
    
    t = time.time() - start_time
    
    print("\n" + "="*80)
    print(" TABLA 1: MATRIZ EJECUTIVA DE DESPEJE (4 CUADRANTES: LUZ VS TIPO DE DIA)")
    print("="*80)
    print(res_ejecutiva.to_string(index=False))
    
    print("\n" + "="*95)
    print(" TABLA 2: MATRIZ FOTOMETRICA INTEGRAL (8 CUADRANTES: SOLAR + CREPUSCULO + DIA)")
    print("="*95)
    print(res_fotometrica.to_string(index=False))
    
    print(f"\n[TIME] Tiempo Total Pandas: {t:.4f} segundos")
    return t

def ejecutar_pyspark(path):
    print("\n[START] [PYSPARK] Ejecutando Consulta 10...")
    start_time = time.time()
    
    spark = SparkSession.builder \
        .appName("C10") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
        
    df = spark.read.parquet(path)
    
    df_clean = df.filter(F.col("Sunrise_Sunset").isNotNull() & F.col("Astronomical_Twilight").isNotNull()) \
                 .withColumn("Start_TS", F.to_timestamp(F.col("Start_Time"))) \
                 .withColumn("End_TS", F.to_timestamp(F.col("End_Time"))) \
                 .filter(F.col("Start_TS").isNotNull() & F.col("End_TS").isNotNull()) \
                 .withColumn("Despeje_Min", (F.unix_timestamp("End_TS") - F.unix_timestamp("Start_TS")) / 60.0) \
                 .filter(F.col("Despeje_Min") >= 0) \
                 .withColumn("Num_Dia", F.dayofweek(F.col("Start_TS"))) \
                 .withColumn("Tipo_Dia", F.when((F.col("Num_Dia") == 1) | (F.col("Num_Dia") == 7), "FIN DE SEMANA").otherwise("DIA HABIL"))
                 
    # 1. Matriz Ejecutiva Spark
    df_ejec = df_clean.groupBy("Sunrise_Sunset", "Tipo_Dia") \
                      .agg(
                          F.count("Start_TS").alias("accidentes_totales"),
                          F.mean("Despeje_Min").alias("despeje_promedio_min")
                      ).orderBy("Sunrise_Sunset", "Tipo_Dia")
                      
    # 2. Matriz Fotométrica Spark
    df_foto = df_clean.groupBy("Sunrise_Sunset", "Astronomical_Twilight", "Tipo_Dia") \
                      .agg(
                          F.count("Start_TS").alias("accidentes_totales"),
                          F.mean("Despeje_Min").alias("despeje_promedio_min")
                      ).orderBy("Sunrise_Sunset", "Astronomical_Twilight", "Tipo_Dia")
                      
    res_ejec = df_ejec.collect()
    res_foto = df_foto.collect()
    
    t = time.time() - start_time
    print("\n--- PYSPARK MATRIZ EJECUTIVA (4 CUADRANTES) ---")
    for r in res_ejec:
        print(f"Luz: {r['Sunrise_Sunset']} | Tipo Dia: {r['Tipo_Dia']} | Siniestros: {r['accidentes_totales']:,} | Despeje: {r['despeje_promedio_min']:.2f} min")
        
    print("\n--- PYSPARK MATRIZ FOTOMETRICA COMPLETA (8 COMBINACIONES) ---")
    for r in res_foto:
        print(f"Solar: {r['Sunrise_Sunset']} | Crepusculo: {r['Astronomical_Twilight']} | Tipo Dia: {r['Tipo_Dia']} | Siniestros: {r['accidentes_totales']:,} | Despeje: {r['despeje_promedio_min']:.2f} min")
        
    print(f"\n[TIME] Tiempo PySpark: {t:.4f} segundos")
    spark.stop()
    return t

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    t_p = ejecutar_pandas(p)
    t_s = ejecutar_pyspark(p)
    print(f"\n[SPEEDUP] SPEEDUP CONSULTA 10: {t_p / t_s:.2f}x")
