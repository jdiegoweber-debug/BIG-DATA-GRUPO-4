# GUION ESTRATÉGICO DE DEFENSA ORAL (12 MINUTOS + 3 MIN PREGUNTAS)
## PROYECTO FINAL: PROCESAMIENTO DISTRIBUIDO Y ANALÍTICA DE ACCIDENTES EN EE. UU. (2016–2023)
### MAESTRÍA EN INTELIGENCIA ARTIFICIAL — ASIGNATURA: BIG DATA — UNIVERSIDAD DE PALERMO (GRUPO 4)

---

## ⏱️ CRONOGRAMA MAESTRO DE TIEMPOS (15 MINUTOS TOTALES)

* **EXPOSITOR 1 (Juan Diego):** Minuto `00:00` a `04:00` (4:00 min) — Slides 1, 2 y 3.
  * *Eje:* Introducción, Arquitectura de almacenamiento Parquet, Ingesta y Análisis Espacial (C3 y C4).
* **EXPOSITOR 2 (Diego):** Minuto `04:00` a `08:00` (4:00 min) — Slides 4 y 5.
  * *Eje:* Corredores viales (C6), Detección de anomalías climáticas (C8) y Velocidad de despeje (C10).
* **EXPOSITOR 3 (Melisa):** Minuto `08:00` a `12:00` (4:00 min) — Slides 6, 7, 8, 9 y 10.
  * *Eje:* Machine Learning distribuido (Consigna 6), Benchmarking Pandas vs. Spark, Conclusiones y Cierre.
* **RONDA DE PREGUNTAS Y DEFENSA ANTE EL DOCENTE:** Minuto `12:00` a `15:00` (3:00 min).

---

## BLOQUE 1: EXPOSITOR 1 (Juan Diego) | [00:00 - 04:00]
### Portada, Arquitectura de Almacenamiento y Análisis Espacial (C3 y C4)

#### 📽️ SLIDE 1: Portada Oficial [00:00 - 00:45]
> *(En pantalla: Slide 1 con título, logo institucional y datos del Grupo 4)*

* **Texto del Expositor:**
  > "Buenas tardes, profesor y compañeros. En nombre del Grupo 4, integrado por Diego, Melisa y quien les habla, Juan Diego Weber, presentamos nuestro proyecto final de Big Data: *'Procesamiento Distribuido y Analítica Predictiva de Accidentes Viales en Estados Unidos (2016–2023)'*.
  >
  > Hemos abordado un universo masivo de más de 7.7 millones de eventos viales mediante un enfoque riguroso de ingeniería de datos. El objetivo central fue evaluar la eficiencia y límites de escalabilidad comparando Pandas vectorizado mononodo frente al procesamiento distribuido con Apache Spark, integrando modelado predictivo con Spark MLlib y resolviendo las 5 consultas analíticas asignadas."

---

#### 📽️ SLIDE 2: Dataset, Arquitectura y Entorno de Cómputo [00:45 - 02:15]
> *(En pantalla: Slide 2 con el gráfico de barras CSV vs. Parquet ZSTD)*

* **Texto del Expositor:**
  > "Para comenzar, observemos el desafío de infraestructura en la diapositiva 2. El dataset en bruto provisto por Moosavi consta de **7,728,394 tuplas y 46 columnas heterogéneas**, pesando 2.87 GB en formato CSV desestructurado y sin tipado nativo.
  >
  > Procesar un CSV de este volumen en memoria genera cuellos de botella severos de I/O y riesgo constante de *Out of Memory*. Por ello, diseñamos un pipeline de reingeniería de almacenamiento columnar convirtiéndolo a **Apache Parquet con compresión ZSTD**, como observan en el gráfico a la derecha. Logramos comprimir el dataset a **491.33 MB**, lo que representa un ahorro de disco y ancho de banda del **82.9%**.
  >
  > Pero la decisión clave de arquitectura fue aplicar un **ordenamiento físico (sorting) por latitud y longitud**. Esto clusteriza geográficamente los bloques internos de Parquet (*Row Groups*), permitiendo aplicar técnicas de *Predicate Pushdown* para descartar a nivel de metadata hasta el 70% de los datos irrelevantes antes de tocar la memoria RAM.
  >
  > El entorno de cómputo se estandarizó sobre **Apache Spark 3.5.0 en PySpark**, OpenJDK 11, Python 3.11 y Pandas 2.2 con soporte de PyArrow, garantizando reproducibilidad tanto en Google Colab como en clúster local."

---

#### 📽️ SLIDE 3: Consultas Analíticas (C3 y C4) [02:15 - 04:00]
> *(En pantalla: Slide 3 con el gráfico dual de Hotspots de C4)*

* **Texto del Expositor:**
  > "Pasando a la analítica espacial, en la Consulta 3 implementamos un filtro paramétrico geodésico utilizando la fórmula del semiverseno (*Haversine*) con un radio de 25 km. Validamos dos polos urbanos: Los Ángeles, con 361,038 siniestros procesados en solo 1.67 segundos en Pandas, y Nueva York con 269,717. En esta consulta de filtro estrecho, la vectorización pura en C de Pandas superó el overhead de serialización de Spark.
  >
  > En la **Consulta 4 (nuestra consigna obligatoria destacada)**, identificamos los *Hotspots* de mayor peligrosidad. Para evitar sesgos, propusimos una **doble perspectiva metodológica**:
  >
  > 1. **Enfoque A (Severidad Crítica Nacional - Severity >= 3):** Discretizamos el territorio en una grilla de 2 decimales (~2.2 km). No buscamos dónde choca más gente levemente, sino dónde se pierden vidas y se saturan terapias intensivas. El ranking nacional coronó al nodo de **Downey en Los Ángeles con 1,619 siniestros graves**, seguido por el cruce del **George Washington Bridge en Nueva York con 1,550**, y tres intercambiadores troncales de las interestatales I-75 e I-85 en **Atlanta** con más de 1,300 choques críticos cada uno.
  >
  > 2. **Enfoque B (Densidad Absoluta Urbana):** Mediante una grilla de mayor radio (~5 km) en California, detectamos el epicentro de congestión en **Downtown Los Ángeles con 50,254 siniestros totales**.
  >
  > Ahora le cedo la palabra a mi compañero Diego, quien profundizará en los corredores viales, anomalías climáticas y la dinámica temporal de despeje."

---

## BLOQUE 2: EXPOSITOR 2 (Diego) | [04:00 - 08:00]
### Corredores Viales (C6), Detección de Anomalías (C8) y Latencia de Despeje (C10)

#### 📽️ SLIDE 4: Consultas Analíticas (C6 y C8) [04:00 - 06:00]
> *(En pantalla: Slide 4 con el gráfico de autopistas de C6)*

* **Texto del Expositor:**
  > "Muchas gracias, Juan Diego. Buenas tardes a todos. Continuando con las consultas asignadas, en la diapositiva 4 abordamos la Consulta 6 y la Consulta 8.
  >
  > En la **Consulta 6**, el reto fue normalizar nombres de calles no estructurados. Construimos una expresión regular compilada para extraer corredores federales canónicos (`I-` y `US-`). Nuevamente, aplicamos una doble métrica de ingeniería:
  >
  > * Si medimos **Tasa de Severidad Crítica Relativa**, las autopistas más peligrosas son corredores rurales de alta velocidad y clima extremo: la **I-72 en Illinois con un alarmante 92.15% de siniestros graves**, y la **I-57 con un 81.15%**.
  > * Pero si medimos **Volumen Bruto de Trauma Vial**, la arteria más letal del país es la **I-95 en la Costa Este con 62,202 siniestros graves** y un tiempo promedio de bloqueo vial de 95.6 minutos, seguida por la I-10 en el sur con más de 44,000.
  >
  > En la **Consulta 8**, buscamos anomalías estadísticas climáticas aplicando Z-score (> 3 sigmas):
  > * Desde la perspectiva de **Volumen de Saturación**, California registró picos históricos absolutos de 2,823 accidentes diarios en enero de 2021.
  > * Pero la perspectiva de **Rareza Meteorológica** reveló que las verdaderas anomalías extremas ocurrieron en Kansas (23.25σ) y Ohio (18.27σ) durante la Tormenta Invernal Elliott de diciembre de 2022, donde vientos de más de 26 mph y visibilidad nula desataron colapsos en cadena con desvíos estadísticos de más de 18 desviaciones estándar."

---

#### 📽️ SLIDE 5: Consultas Analíticas (C10: Velocidad de Despeje) [06:00 - 08:00]
> *(En pantalla: Slide 5 con el gráfico de barras fotométrico de C10)*

* **Texto del Expositor:**
  > "En la diapositiva 5 presentamos la **Consulta 10**, una de las piezas analíticas más reveladoras sobre los 7.72 millones de tuplas. Analizamos el tiempo medio de despeje vial cruzando condiciones de iluminación y tipo de día.
  >
  > Construimos dos matrices:
  > 1. Una **Matriz Ejecutiva de 4 cuadrantes**, donde el día hábil promedia 336.5 minutos (~5.6 horas) y la noche de fin de semana salta a 720.1 minutos (~12 horas).
  > 2. Pero al profundizar en la **Matriz Fotométrica Integral de 8 cuadrantes**, desglosando la luz solar frente al Crepúsculo Astronómico (*Astronomical Twilight*), descubrimos el verdadero comportamiento del sistema:
  >
  > Mientras que en el crepúsculo vespertino la luz residual amortigua la latencia a 345 minutos, en **Noche Cerrada de Fin de Semana la duración media escala a 772.75 minutos, es decir, ¡casi 13 horas de bloqueo continuo!**
  >
  > Esto representa un **incremento crítico del +130.4% en la congestión vial**. La causa no es meramente meteorológica: responde a un fallo estructural de logística pública: la falta de disponibilidad de grúas de gran porte fuera de horario administrativo y la menor dotación de cuadrillas forenses y viales durante las madrugadas del fin de semana.
  >
  > A continuación, Melisa expondrá el modelado predictivo con Spark MLlib y el benchmarking de rendimiento."

---

## BLOQUE 3: EXPOSITOR 3 (Melisa) | [08:00 - 12:00]
### Machine Learning con Spark MLlib (Consigna 6), Benchmarking, Conclusiones y Cierre

#### 📽️ SLIDE 6: Planteo del Problema de ML (Justificación y Pipeline) [08:00 - 09:15]
> *(En pantalla: Slide 6 con la arquitectura en dos columnas del pipeline de ML)*

* **Texto del Expositor:**
  > "Muchas gracias, Diego. Buenas tardes. En la Consigna 6 abordamos el modelado predictivo a escala masiva mediante **Spark MLlib**.
  >
  > 1. **Justificación del Problema:** Definimos como variable objetivo a predecir la **Severidad vial (Niveles 1 a 4)**. En Sistemas Inteligentes de Transporte (ITS), conocer la severidad probable en los primeros segundos de un siniestro permite despachar unidades de soporte vital y alterar la señalización antes de que se propague el colapso de la red.
  >
  > 2. **Prevención de Data Leakage:** Un error común en este dataset es incluir variables post-evento como la duración del despeje o la distancia de congestión. Nuestro pipeline las excluyó estrictamente, garantizando que el modelo opere **únicamente con 54 factores exógenos y ambientales preexistentes al choque**.
  >
  > 3. **Pipeline Distribuido pyspark.ml:** Diseñamos un flujo que imputa nulos, indexa las variables categóricas con `StringIndexer`, codifica estados geográficos con `OneHotEncoderEstimator`, extrae componentes temporales cíclicas y empaqueta las 13 variables de infraestructura en un vector de características mediante `VectorAssembler`.
  >
  > Elegimos un clasificador supervisado `DecisionTreeClassifier` configurando el hiperparámetro `maxBins=60` para capturar con precisión la cardinalidad de los 49 estados."

---

#### 📽️ SLIDE 7: Resultados Experimentales y Evaluación del Modelo [09:15 - 10:15]
> *(En pantalla: Slide 7 con el gráfico de métricas y feature importances)*

* **Texto del Expositor:**
  > "En la diapositiva 7 observamos la evaluación sobre una partición rigurosa del **80% para entrenamiento y 20% para test**, procesando millones de registros en 245 segundos.
  >
  > Los resultados alcanzados superan ampliamente el baseline mayoritario del 73.2%:
  > * **Accuracy global del 79.81%**
  > * **F1-Score Ponderado de 0.7138**, lo cual confirma la robustez del modelo frente al severo desbalanceo de clases intrínseco de los accidentes viales.
  >
  > Al inspeccionar los nodos raíz del árbol de decisión, confirmamos nuestra hipótesis de investigación: las variables con mayor ganancia de información son la **Visibilidad inferior a 3 millas** y la presencia de **Intercambiadores viales (`Junction == True`)**. Cuando ambas condiciones convergen, la probabilidad de que un siniestro sea clasificado en Severidad 3 o 4 se cuadruplica."

---

#### 📽️ SLIDE 8: Benchmarking Pandas vs. Spark y Discusión del Speedup [10:15 - 11:15]
> *(En pantalla: Slide 8 con la tabla de tiempos y el gráfico de barras logarítmico)*

* **Texto del Expositor:**
  > "En la diapositiva 8 presentamos la comparativa rigurosa de tiempos de ejecución entre Pandas y PySpark en las 5 consultas.
  >
  > La discusión técnica arroja una lección fundamental de arquitectura:
  > * En consultas como C3, C4 y C6, caracterizadas por proyecciones de pocas columnas y filtros espaciales simples, **Pandas vectorizado en memoria mononodo es entre 3 y 5 veces más veloz**. Esto se debe a que opera directamente en C y PyArrow con cero latencia de serialización, mientras que Spark incurre en la sobrecarga de inicialización de la JVM, DAG y Py4J.
  > * Sin embargo, en la **Consulta 10**, con agregaciones multinivel complejas sobre los 7.72M de registros, **PySpark se impone con un Speedup de 1.65x**, ejecutando en 18.25 segundos frente a los 30.04 de Pandas.
  >
  > Esto demuestra que Spark no busca competir en latencias de milisegundos en tareas triviales, sino brindar **escalabilidad horizontal y capacidad *out-of-core*** cuando los datos no caben en la RAM física de una sola máquina."

---

#### 📽️ SLIDES 9 Y 10: Conclusiones Técnicas, Cierre y Preguntas [11:15 - 12:00]
> *(En pantalla: Slide 9 y pase a Slide 10)*

* **Texto del Expositor:**
  > "Para concluir, destacamos cuatro lecciones centrales:
  > 1. La reingeniería a **Parquet con compresión ZSTD y ordenamiento geodésico** ahorró un 83% de almacenamiento y eliminó hasta un 70% de lecturas redundantes en disco.
  > 2. **Pandas y Spark son complementarios**, no excluyentes: Pandas para analítica ágil mononodo; Spark para agregaciones masivas y pipelines tolerantes a fallos.
  > 3. La **doble perspectiva metodológica** es obligatoria en Big Data para no confundir densidad demográfica con riesgo real.
  > 4. Las casi **13 horas de bloqueo en fines de semana** y la precisión del **79.8% de nuestro modelo ML** aportan herramientas concretas para el diseño de políticas públicas viales e ITS.
  >
  > *(Pase a Slide 10)*
  > Agradecemos al cuerpo docente de la Universidad de Palermo. Dejamos abiertos los 3 minutos restantes para sus preguntas y comentarios."

---

## BLOQUE 4: BANCO ESTRATÉGICO DE PREGUNTAS DEL DOCENTE [12:00 - 15:00]

### Pregunta 1: "¿Por qué Pandas fue más rápido que Spark en casi todas las consultas?"
* **Quién responde:** **Juan Diego** o **Melisa**.
* **Respuesta técnica contundente:**
  > *"Excelente pregunta, profesor. La diferencia radica en la arquitectura de cómputo y el tamaño del dataset. Con 491 MB en formato Parquet, el dataset cabe íntegramente en la memoria RAM del nodo. Pandas, apalancado en el backend columnar de C++ y PyArrow, procesa punteros en memoria contigua sin intermediarios. Apache Spark, en cambio, tiene un costo fijo ineludible: la creación del SparkContext, el armado del DAG lógico, la traducción al plan físico en Catalyst y la comunicación entre la JVM y Python mediante sockets IPC (Py4J). Ese overhead representa unos 6 a 8 segundos basales. Sin embargo, si este dataset tuviera 700 millones de filas en lugar de 7 millones, Pandas colapsaría por OOM (Out of Memory), mientras que Spark continuaría procesando mediante particionado distribuido y spilling en disco."*

### Pregunta 2: "¿Por qué utilizaron una grilla de 2 decimales en la Consulta 4 en lugar de calcular Haversine contra todos los puntos?"
* **Quién responde:** **Juan Diego**.
* **Respuesta técnica contundente:**
  > *"Calcular la matriz completa de distancias de Haversine para 7.72 millones de tuplas tiene una complejidad cuadrática de $O(N^2)$, lo que implica evaluar más de 59 billones de combinaciones de distancias geodésicas. Esto es computacionalmente intratable en un entorno interactivo y generaría un desbordamiento inmediato de memoria. La discretización por truncamiento a 2 decimales proyecta el espacio continuo en una cuadrícula celular donde 0.01° de latitud equivale a ~1.11 km, definiendo una celda cuadrada de influencia de ~2.2 km. Esto redujo el problema de $O(N^2)$ a un agrupamiento con hash de complejidad $O(N)$, completándose en apenas 1.73 segundos sin perder representatividad espacial."*

### Pregunta 3: "¿Cómo garantizaron que el modelo de Machine Learning no tuviera Data Leakage?"
* **Quién responde:** **Melisa**.
* **Respuesta técnica contundente:**
  > *"Auditamos minuciosamente el catálogo de 46 atributos del dataset original. Variables como 'End_Time', la duración calculada del siniestro, la longitud de la congestión formada ('Distance(mi)') y el texto libre del parte de tráfico solo existen una vez que el incidente ya ocurrió y fue intervenido por la autoridad vial. Si las incluíamos, el modelo predeciría con un Accuracy engañoso superior al 95%, pero sería completamente inútil en un entorno de producción en tiempo real. Por ello, restringimos el vector de características a variables estrictamente previas al choque: momento temporal (hora, mes, día), meteorología local en la estación más cercana (temperatura, visibilidad, lluvia, viento) e infraestructura física fija (semáforos, cruces y bifurcaciones)."*

### Pregunta 4: "¿Por qué en la Consulta 10 hay tanta diferencia entre el día hábil y la noche de fin de semana?"
* **Quién responde:** **Diego**.
* **Respuesta técnica contundente:**
  > *"El análisis fotométrico cruzado demostró que no se trata únicamente de un factor de visibilidad, ya que en el crepúsculo astronómico el despeje se mantiene en 345 minutos. La causa principal de que en la noche de fin de semana trepe a 772.75 minutos (~12.88 horas) reside en la disponibilidad de recursos operativos: los departamentos de transporte estatales (DOT) y los servicios de grúas pesadas operan con guardias mínimas durante las madrugadas no laborables. Además, los siniestros nocturnos de fin de semana presentan mayor tasa de colisiones a alta velocidad y consumos de sustancias, lo que exige peritajes policiales y judiciales en el lugar del hecho antes de poder liberar la calzada."*
