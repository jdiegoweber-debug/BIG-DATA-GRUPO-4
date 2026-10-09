"""
MIA - Big Data - Universidad de Palermo
GRUPO 4 - Consigna 6 (Opcion A): Pipeline de Clasificacion Distribuida con Spark MLlib
Objetivo: Predecir la severidad del accidente (Severity) utilizando variables meteorologicas y geograficas.
"""
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def entrenar_pipeline_ml(parquet_path):
    print("\n==================================================")
    print("[ML] INICIANDO PIPELINE DE MACHINE LEARNING (SPARK MLlib)")
    print("==================================================")
    start_time = time.time()
    
    # 1. Inicializar sesion de Spark optimizada para ML
    spark = SparkSession.builder \
        .appName("MIA_Grupo4_ML") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
        
    print("[WAIT] Cargando y preprocesando datos...")
    df = spark.read.parquet(parquet_path)
    
    # Seleccionamos variables predictoras clave y removemos nulos estructurales
    # Ajustamos la severidad para que empiece en 0 (Spark ML prefiere etiquetas indexadas de 0 a N-1)
    df_ml = df.select(
        "Severity", "Start_Lat", "Start_Lng", "Temperature(F)", 
        "Humidity(%)", "Visibility(mi)", "Sunrise_Sunset", "State"
    ).dropna() \
     .withColumn("label", F.col("Severity") - 1)
    
    # 2. Ingenieria de Caracteristicas Distribuidas (Pipeline Stages)
    # Convertir variables categoricas (Strings) a Indices Numericos
    indexer_sunset = StringIndexer(inputCol="Sunrise_Sunset", outputCol="Sunset_Index")
    indexer_state = StringIndexer(inputCol="State", outputCol="State_Index")
    
    # Aplicar One-Hot Encoding a las variables indexadas
    encoder = OneHotEncoder(
        inputCols=["Sunset_Index", "State_Index"],
        outputCols=["Sunset_OHE", "State_OHE"]
    )
    
    # Ensamblar todos los vectores en una unica columna de caracteristicas de entrada ('features')
    columnas_features = [
        "Start_Lat", "Start_Lng", "Temperature(F)", 
        "Humidity(%)", "Visibility(mi)", "Sunset_OHE", "State_OHE"
    ]
    assembler = VectorAssembler(inputCols=columnas_features, outputCol="features")
    
    # 3. Definir el Clasificador (Arbol de Decision Multiclase)
    classifier = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxBins=60)
    
    # 4. Construir el Pipeline Integrado de Spark
    pipeline = Pipeline(stages=[indexer_sunset, indexer_state, encoder, assembler, classifier])
    
    # 5. Division del Dataset (Train/Test Split de forma distribuida y reproducible)
    print("[FILE] Dividiendo dataset en 80% Entrenamiento y 20% Testeo...")
    train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)
    
    # 6. Entrenamiento del Modelo (Ajuste del Pipeline)
    print("[TRAIN] Entrenando el modelo de clasificacion distribuida... (Esto puede tomar un momento)")
    model_pipeline = pipeline.fit(train_data)
    
    # 7. Inferencia sobre el conjunto de prueba
    print("[PRED] Realizando predicciones sobre el conjunto de testeo...")
    predicciones = model_pipeline.transform(test_data)
    
    # 8. Evaluacion de Metricas de Calidad Academica
    evaluador_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluador_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    
    accuracy = evaluador_accuracy.evaluate(predicciones)
    f1_score = evaluador_f1.evaluate(predicciones)
    
    end_time = time.time()
    print("\n==================================================")
    print("[RESULTS] RESULTADOS DEL MODELO DE CLASIFICACION")
    print("==================================================")
    print(f"[OK] Exactitud Global (Accuracy): {accuracy * 100:.2f}%")
    print(f"[OK] Metrica F1-Score: {f1_score:.4f}")
    print(f"[TIME] Tiempo total del proceso de ML: {end_time - start_time:.2f} segundos")
    print("==================================================")
    
    # Mostrar una muestra de control de las predicciones vs las etiquetas reales
    predicciones.select("Severity", "prediction", "features").show(5, truncate=False)
    
    spark.stop()

if __name__ == "__main__":
    p = os.path.join("data", "raw", "us_accidents.parquet")
    if not os.path.exists(p):
        print("[ERROR] Archivo Parquet no encontrado.")
    else:
        entrenar_pipeline_ml(p)
