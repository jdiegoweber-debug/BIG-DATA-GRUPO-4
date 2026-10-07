"""
MIA - Big Data — Universidad de Palermo
GRUPO 4 - Conversión e Indexación Óptima de CSV a Parquet (A prueba de fallos de formatos)
"""
import os
import time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

def csv_to_parquet_indexed_grupo4():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
    csv_path = os.path.join(BASE_DIR, "data", "raw", "us_accidents.csv")
    parquet_path = os.path.join(BASE_DIR, "data", "raw", "us_accidents.parquet")
    
    print("⏳ Iniciando la conversión e indexación interna del archivo Parquet...")
    if not os.path.exists(csv_path):
        print(f"❌ Error: No se encontró el archivo {csv_path}.")
        return
        
    start_time = time.time()
    
    # 1. MAPEADO SEGURO DE TIPOS ANTE VARIACIONES DEL CSV
    # Forzamos numéricos clave y dejamos que las fechas y categorías sean strings estables para evitar caídas
    column_types = {
        "Start_Lat": pa.float64(),
        "Start_Lng": pa.float64(),
        "End_Lat": pa.float64(),
        "End_Lng": pa.float64(),
        "Severity": pa.int64(),
        "Number": pa.float64(),
        "Wind_Speed(mph)": pa.float64(),
        "Precipitation(in)": pa.float64(),
        "Visibility(mi)": pa.float64(),
        "Temperature(F)": pa.float64(),
        "Humidity(%)": pa.float64(),
        "Pressure(in)": pa.float64(),
        "Start_Time": pa.string(),  # Evita el error del bloque 68 con timestamps extraños
        "End_Time": pa.string(),    # Evita posibles errores futuros en End_Time
        "Weather_Timestamp": pa.string()
    }
    
    read_options = pv.ReadOptions(block_size=20_000_000) 
    convert_options = pv.ConvertOptions(column_types=column_types, strings_can_be_null=True)
    parse_options = pv.ParseOptions(invalid_row_handler=lambda row: "skip")
    
    reader = pv.open_csv(
        csv_path, 
        read_options=read_options, 
        parse_options=parse_options,
        convert_options=convert_options
    )
    
    parquet_writer = None
    chunk_count = 0
    
    print("🛠️ Inyectando índices físicos por bloques (Sorting + Row Group Metadata)...")
    
    for chunk in reader:
        table = pa.Table.from_batches([chunk])
        chunk_count += 1
        
        # 2. CONSTRUCCIÓN DEL ÍNDICE INTERNO (Orden Jerárquico para el Grupo 4)
        sort_keys = []
        for col in ["State", "Start_Lat", "Start_Lng", "Street"]:
            if col in table.column_names:
                sort_keys.append((col, "ascending"))
        
        if sort_keys:
            indices = pc.select_k_unstable(table, sort_keys=sort_keys, k=len(table))
            table = table.take(indices)
        
        # 3. ESCRITURA E INYECCIÓN DE METADATOS DE BÚSQUEDA RÁPIDA
        if parquet_writer is None:
            writer_properties = {
                'compression': 'zstd',          
                'use_dictionary': True,         
                'write_statistics': True        
            }
            
            parquet_writer = pq.ParquetWriter(
                parquet_path, 
                table.schema, 
                **writer_properties
            )
            
        parquet_writer.write_table(table, row_group_size=100_000)
        print(f"✅ Bloque {chunk_count} procesado, ordenado e indexado en disco...")
        
    if parquet_writer:
        parquet_writer.close()

    end_time = time.time()
    elapsed = end_time - start_time
    
    size_csv = os.path.getsize(csv_path) / (1024**3)
    size_pq = os.path.getsize(parquet_path) / (1024**2)
    
    print(f"\n🎉 ¡Proceso completado con éxito en {elapsed:.2f} segundos!")
    print(f"📦 Archivo Parquet indexado generado en: {parquet_path}")
    print(f"📏 Tamaño del CSV original: {size_csv:.2f} GB")
    print(f"⚡ Tamaño del Parquet indexado y comprimido: {size_pq:.2f} MB")

if __name__ == "__main__":
    csv_to_parquet_indexed_grupo4()
