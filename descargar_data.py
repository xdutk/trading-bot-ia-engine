import ccxt
import pandas as pd
import os
import time
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
# Fecha de inicio: Cuanto más atrás, más tarda, pero más aprende la IA.
# 2020 es un buen punto (incluye el Bull Run anterior, el Bear Market y el actual).
FECHA_INICIO = "2020-01-01 00:00:00" 
TIMEFRAME = '5m'

# Lista del Top 20 (Manual para asegurar calidad)
TOP20_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'DOT/USDT', 
    'MATIC/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT', 
    'UNI/USDT', 'XLM/USDT', 'ETC/USDT', 'FIL/USDT', 'NEAR/USDT'
]

# Rutas de salida (Las mismas que definimos antes)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_REFERENCIA = os.path.join(BASE_DIR, 'DATOS_ENTRADA', 'REFERENCIA_TOP20')
DIR_OPERAR = os.path.join(BASE_DIR, 'DATOS_ENTRADA', 'ACTIVOS_A_OPERAR')

# Crear carpetas si no existen
os.makedirs(DIR_REFERENCIA, exist_ok=True)
os.makedirs(DIR_OPERAR, exist_ok=True)

def descargar_historial(symbol, start_date_str):
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Convertir fecha a timestamp (ms)
    since = exchange.parse8601(start_date_str)
    
    all_ohlcv = []
    limit = 1000 # Límite máximo de Binance por petición
    
    print(f"⬇️  Iniciando descarga de {symbol} desde {start_date_str}...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=limit)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Actualizar 'since' al tiempo de la última vela + 1 timeframe (5 min = 300000ms)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 300000 
            
            # Feedback visual
            fecha_actual = datetime.fromtimestamp(last_timestamp / 1000)
            print(f"   ... descargado hasta {fecha_actual} ({len(all_ohlcv)} velas)")
            
            # Si llegamos al presente, paramos
            now = exchange.milliseconds()
            if since > now:
                break
                
            # Respetar Rate Limits (Pausa pequeña)
            # time.sleep(0.1) # ccxt maneja esto si enableRateLimit=True, pero por seguridad
            
        except Exception as e:
            print(f"❌ Error descargando: {e}")
            time.sleep(5) # Esperar y reintentar
            continue

    if not all_ohlcv:
        return None

    # Convertir a DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    # Binance devuelve el tiempo en ms, lo dejamos así o lo convertimos. 
    # Para consistencia con nuestros scripts, lo dejamos crudo o lo convertimos.
    # Vamos a dejarlo crudo (int) y que pipeline_runner lo convierta a datetime.
    
    return df

def ejecutar_descarga_masiva():
    print("=========================================================")
    print("📥 DESCARGADOR MASIVO DE BINANCE (CCXT)")
    print("=========================================================\n")
    
    for symbol in TOP20_SYMBOLS:
        nombre_limpio = symbol.replace('/', '') # BTC/USDT -> BTCUSDT
        filename = f"{nombre_limpio}.csv"
        
        # 1. Descargar
        df = descargar_historial(symbol, FECHA_INICIO)
        
        if df is not None:
            # 2. Guardar en REFERENCIA (Top 20)
            path_ref = os.path.join(DIR_REFERENCIA, filename)
            df.to_csv(path_ref, index=False)
            print(f"✅ Guardado en REFERENCIA: {filename}")
            
            # 3. Guardar en ACTIVOS_A_OPERAR (Para operar/entrenar)
            # (Aquí podrías filtrar si solo quieres operar algunas, pero para el test bajamos todo)
            path_op = os.path.join(DIR_OPERAR, filename)
            df.to_csv(path_op, index=False)
            print(f"✅ Guardado en OPERAR: {filename}")
            
            print("-" * 40)
        else:
            print(f"⚠️ No se pudieron bajar datos para {symbol}")

    print("\n🎉 ¡Descarga completada!")

if __name__ == "__main__":
    ejecutar_descarga_masiva()