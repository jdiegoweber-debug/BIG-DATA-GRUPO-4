import os
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def csv_to_parquet():
    raw_dir = os.path.join("data", "raw")
    csv_path = os.path.join(raw_dir, "us_accidents.csv")
    parquet_path = os.path.join(raw_dir, "us_accidents.parquet")
    
    print("⏳ Iniciando la conversión de CSV a Parquet...")
    if not os.path.exists(csv_path):
        print(f"❌ Error: No se encontró el archivo {csv_path}.")
        print("💡 Verificá que el archivo esté dentro de 'data/raw/' y se llame exactamente 'us_accidents.csv'")
        return
        
    start_time = time.time()
    chunk_size = 500_000
    parquet_writer = None
    chunk_count = 0
    
    # Leemos el CSV en bloques
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        # Convertimos el bloque de Pandas a una Tabla de PyArrow
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        
        # En el primer bloque inicializamos el escritor con el esquema de datos
        if parquet_writer == None:
            parquet_writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
            
        # Escribimos la tabla en el archivo Parquet
        parquet_writer.write_table(table)
        chunk_count += 1
        print(f"✅ Procesados {chunk_count * chunk_size:,} registros...")
        
    # Cerramos el escritor al finalizar
    if parquet_writer:
        parquet_writer.close()

    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n🎉 ¡Conversión finalizada con éxito en {elapsed:.2f} segundos!")
    print(f"📦 Archivo Parquet generado en: {parquet_path}")
    print(f"📏 Tamaño del CSV original: {os.path.getsize(csv_path) / (1024**3):.2f} GB")
    print(f"⚡ Tamaño del Parquet comprimido: {os.path.getsize(parquet_path) / (1024**2):.2f} MB")

if __name__ == "__main__":
    csv_to_parquet()
