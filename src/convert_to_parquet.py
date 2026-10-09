import os
import time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

# Definición explícita de tipos de datos para lectura por bloques robusta
CSV_DTYPES = {
    'ID': 'string',
    'Source': 'string',
    'Severity': 'int64',
    'Start_Time': 'string',
    'End_Time': 'string',
    'Start_Lat': 'float64',
    'Start_Lng': 'float64',
    'End_Lat': 'float64',
    'End_Lng': 'float64',
    'Distance(mi)': 'float64',
    'Description': 'string',
    'Street': 'string',
    'City': 'string',
    'County': 'string',
    'State': 'string',
    'Zipcode': 'string',
    'Country': 'string',
    'Timezone': 'string',
    'Airport_Code': 'string',
    'Weather_Timestamp': 'string',
    'Temperature(F)': 'float64',
    'Wind_Chill(F)': 'float64',
    'Humidity(%)': 'float64',
    'Pressure(in)': 'float64',
    'Visibility(mi)': 'float64',
    'Wind_Direction': 'string',
    'Wind_Speed(mph)': 'float64',
    'Precipitation(in)': 'float64',
    'Weather_Condition': 'string',
    'Amenity': 'boolean',
    'Bump': 'boolean',
    'Crossing': 'boolean',
    'Give_Way': 'boolean',
    'Junction': 'boolean',
    'No_Exit': 'boolean',
    'Railway': 'boolean',
    'Roundabout': 'boolean',
    'Station': 'boolean',
    'Stop': 'boolean',
    'Traffic_Calming': 'boolean',
    'Traffic_Signal': 'boolean',
    'Turning_Loop': 'boolean',
    'Sunrise_Sunset': 'string',
    'Civil_Twilight': 'string',
    'Nautical_Twilight': 'string',
    'Astronomical_Twilight': 'string',
}

def csv_to_parquet(csv_path=None, parquet_path=None, chunk_size=250_000):
    """
    Convierte us_accidents.csv a formato Parquet optimizado con índices físicos internos,
    compresión ZSTD y estadísticas min/max por bloque.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else os.path.abspath(".")
        csv_path = os.path.join(base_dir, "data", "raw", "us_accidents.csv")
    if parquet_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else os.path.abspath(".")
        parquet_path = os.path.join(base_dir, "data", "raw", "us_accidents.parquet")

    print("[INFO] Iniciando la conversion e indexacion interna del archivo Parquet...")
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontro el archivo {csv_path}.")
        print("[INFO] Verifica que el archivo este dentro de 'data/raw/' y se llame exactamente 'us_accidents.csv'")
        return False
        
    start_time = time.time()
    parquet_writer = None
    processed_rows = 0
    
    print(f"[INFO] Leyendo CSV en chunks de {chunk_size:,} filas...")
    print("[INFO] Inyectando indices fisicos por bloques (Sorting geo + Metadatos ZSTD + Diccionario)...")
    
    # Creamos directorio destino si no existe
    os.makedirs(os.path.dirname(os.path.abspath(parquet_path)), exist_ok=True)
    
    # Archivo temporal para escritura atómica
    temp_parquet_path = parquet_path + ".tmp"
    if os.path.exists(temp_parquet_path):
        try:
            os.remove(temp_parquet_path)
        except OSError:
            pass

    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, dtype=CSV_DTYPES, low_memory=False), 1):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            
            # Ordenamiento espacial interno por bloque para optimizar consultas 3 y 4
            # Usamos sort_indices con null_placement='at_end' para NO perder filas con NaN
            sort_keys = []
            for col in ["Start_Lat", "Start_Lng"]:
                if col in table.column_names:
                    sort_keys.append((col, "ascending"))
            
            if sort_keys:
                indices = pc.sort_indices(table, sort_keys=sort_keys, null_placement='at_end')
                table = table.take(indices)
            
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    temp_parquet_path, 
                    table.schema, 
                    compression='zstd',
                    use_dictionary=True,
                    write_statistics=True
                )
                
            parquet_writer.write_table(table)
            processed_rows += len(table)
            print(f"[OK] Chunk #{chunk_idx} procesado ({processed_rows:,} filas acumuladas)...")
            
        if parquet_writer:
            parquet_writer.close()
            parquet_writer = None
            
        # Reemplazar archivo final
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
        os.rename(temp_parquet_path, parquet_path)
        
    except Exception as e:
        if parquet_writer:
            parquet_writer.close()
        if os.path.exists(temp_parquet_path):
            try:
                os.remove(temp_parquet_path)
            except OSError:
                pass
        print(f"[ERROR] Error durante la conversion a Parquet: {e}")
        raise e

    elapsed = time.time() - start_time
    size_csv = os.path.getsize(csv_path) / (1024**3)
    size_pq = os.path.getsize(parquet_path) / (1024**2)
    
    print(f"\n[OK] Proceso completado con exito en {elapsed:.2f} segundos!")
    print(f"[INFO] Total filas convertidas: {processed_rows:,}")
    print(f"[INFO] Archivo Parquet indexado generado en: {parquet_path}")
    print(f"[INFO] Tamanio del CSV original: {size_csv:.2f} GB")
    print(f"[INFO] Tamanio del Parquet indexado y comprimido: {size_pq:.2f} MB")
    return True

if __name__ == "__main__":
    csv_to_parquet()