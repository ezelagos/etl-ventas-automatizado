import os                       
import pandas as pd             
import numpy as np              
import yaml                     
import logging                  
from datetime import datetime  
import unicodedata              

#crear nombre dinamico del archivo log
fecha_hoy = datetime.today().strftime('%d-%m-%Y')
log_filename = f'logs/etl_{fecha_hoy}.log'

#configurar el logging (archivo + consola)
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

#funcion para leer el archivo etl_config.yaml
def cargar_configuracion(ruta_config):
    try:
        with open(ruta_config, 'r') as file:
            config = yaml.safe_load(file)
            logging.info("✅ Configuración cargada correctamente.")
            return config
    except Exception as e:
        logging.critical(f"❌ Error cargando configuración: {e}")
        raise

#funcion para remover tildes de un string
def remover_tildes(texto):
    if isinstance(texto, str):
        return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    return texto

#funcion para leer todos los archivos CSV
def leer_archivos_csv(ruta_raw):
    archivos = [f for f in os.listdir(ruta_raw) if f.endswith(".csv")]
    dataframe = []
    for archivo in archivos:
        ruta = os.path.join(ruta_raw, archivo)
        try:
            df = pd.read_csv(ruta, encoding='latin1')
            dataframe.append(df)
            logging.info(f"📄 Archivo cargado: {archivo}")
        except Exception as e:
            logging.error(f"⚠️ Error al leer {archivo}: {e}")
    if not dataframe:
        logging.warning("⚠️ No se encontraron archivos válidos en la carpeta raw.")
    return pd.concat(dataframe, ignore_index=True) if dataframe else pd.DataFrame()

def main():
    logging.info("🚀 Inicio del proceso ETL")
    config = cargar_configuracion("etl_config.yaml")

    logging.info("📥 Leyendo archivos CSV crudos...")
    df = leer_archivos_csv(config["raw_data_path"])

    if df.empty:
        logging.warning("⚠️ DataFrame vacío. Proceso finalizado sin datos.")
        return

    logging.info("🧹 Filtrando columnas necesarias...")
    columnas_utiles = config["columns_to_keep"]
    try:
        df = df[columnas_utiles]
        logging.info("✅ Columnas filtradas correctamente.")
    except Exception as e:
        logging.error(f"❌ Error al filtrar columnas: {e}")
        return

    df = normalizar_texto(df, ["producto", "vendedor", "sucursal"])
    df = tipificar_campos(df)
    df = limpiar_duplicados_y_nulos(df)
    df = validar_reglas_negocio(df)
    df = estandarizar_categorias(df)
    guardar_procesado(df, config["processed_data_path"])
    df = crear_monto_total(df)
    resumen = resumen_diario_por_sucursal(df)
    guardar_resumen_diario(resumen, config["report_path"])

    top_prod = top_productos(df, n=5)
    top_vend = top_vendedores(df, n=5)

    logging.debug("📈 Vista resumen diario (primeras filas):\n" + resumen.head().to_string())
    logging.debug("🥇 Top productos:\n" + top_prod.to_string(index=False))
    logging.debug("🧑‍💼 Top vendedores:\n" + top_vend.to_string(index=False))
    logging.debug("🔍 Vista previa de los datos:")
    logging.debug(f"\n{df.head()}")
    logging.info("✅ ETL finalizado correctamente. Próxima fase: limpieza.")

def normalizar_texto(df, cols_texto):
    for c in cols_texto:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"None": np.nan, "nan": np.nan, "Nan": np.nan})
            df[c] = df[c].apply(remover_tildes)  # Aplicar normalización
    logging.info("🧽 Normalización básica de texto completada.")
    return df

def tipificar_campos(df):
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df["precio_unitario"] = pd.to_numeric(df["precio_unitario"], errors="coerce")
    logging.info("🧱 Tipificación: fechas y numéricos aplicados.")
    return df

def limpiar_duplicados_y_nulos(df):
    antes = len(df)
    df = df.drop_duplicates()
    dup = antes - len(df)
    if dup > 0:
        logging.warning(f"♻️ Duplicados eliminados: {dup}")
    criticos = ["fecha", "producto", "cantidad", "precio_unitario"]
    nulos_antes = df[criticos].isna().sum().sum()
    df = df.dropna(subset=criticos)
    nulos_despues = df[criticos].isna().sum().sum()
    if nulos_antes > nulos_despues:
        logging.warning(f"🚫 Filas eliminadas por nulos críticos: {nulos_antes - nulos_despues}")
    return df

def validar_reglas_negocio(df, ruta_rechazos="data/processed/rechazados.csv"):
    hoy = pd.to_datetime(datetime.today().date())
    mask_futuras = df["fecha"] > hoy
    mask_precio_invalido = (df["precio_unitario"] <= 0) | (df["precio_unitario"].isna())
    mask_cantidad_invalida = (df["cantidad"] <= 0) | (df["cantidad"].isna())
    mask_fecha_invalida = df["fecha"].isna()
    rechazados = df[mask_futuras | mask_precio_invalido | mask_cantidad_invalida | mask_fecha_invalida].copy()
    df_valido = df.drop(rechazados.index)
    if not rechazados.empty:
        os.makedirs(os.path.dirname(ruta_rechazos), exist_ok=True)
        modo = "a" if os.path.exists(ruta_rechazos) else "w"
        encabezado = not os.path.exists(ruta_rechazos)
        rechazados.to_csv(ruta_rechazos, index=False, mode=modo, header=encabezado)
        logging.error(f"🧯 Registros rechazados por reglas de negocio: {len(rechazados)}. Guardados en {ruta_rechazos}")
    logging.info(f"✅ Registros válidos tras reglas de negocio: {len(df_valido)}")
    return df_valido

def estandarizar_categorias(df):
    if "sucursal" in df.columns:
        df["sucursal"] = df["sucursal"].str.title()
    if "producto" in df.columns:
        df["producto"] = df["producto"].str.title()
    if "vendedor" in df.columns:
        df["vendedor"] = df["vendedor"].str.title()
    logging.info("🎛️ Estandarización de categorías aplicada.")
    return df

def guardar_procesado(df, ruta_processed):
    os.makedirs(ruta_processed, exist_ok=True)
    nombre = f"ventas_limpias_{datetime.today().strftime('%d-%m-%Y')}.csv"
    destino = os.path.join(ruta_processed, nombre)
    df.to_csv(destino, index=False)
    logging.info(f"💾 Datos limpios guardados en: {destino}")

def crear_monto_total(df):
    df = df.copy()
    df["monto_total"] = df["cantidad"] * df["precio_unitario"]
    logging.info("➕ Columna monto_total creada.")
    return df

def resumen_diario_por_sucursal(df):
    base = df.copy()
    base["fecha"] = pd.to_datetime(base["fecha"]).dt.date
    g = base.groupby(["fecha", "sucursal"], dropna=False).agg(
        ventas_totales=("monto_total", "sum"),
        unidades=("cantidad", "sum"),
        tickets=("producto", "count")
    ).reset_index()
    g["ticket_promedio"] = (g["ventas_totales"] / g["tickets"]).replace([np.inf, -np.inf], np.nan)
    logging.info("📊 Resumen diario por sucursal generado.")
    return g

def top_productos(df, n=5):
    g = (df.groupby("producto")["monto_total"].sum().sort_values(ascending=False).head(n).reset_index())
    logging.info(f"🏆 Top {n} productos calculado.")
    return g

def top_vendedores(df, n=5):
    df2 = df.copy()
    df2["vendedor"] = df2["vendedor"].fillna("Sin Vendedor")
    g = (df2.groupby("vendedor")["monto_total"].sum().sort_values().head(n).reset_index())
    logging.info(f"🧑‍💼 Top {n} vendedores calculado.")
    return g

def guardar_resumen_diario(df_resumen, report_path):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    modo = "a" if os.path.exists(report_path) else "w"
    header = not os.path.exists(report_path)
    df_resumen.to_csv(report_path, index=False, mode=modo, header=header)
    logging.info(f"📤 Resumen diario anexado en: {report_path}")

if __name__ == "__main__":
    main()
