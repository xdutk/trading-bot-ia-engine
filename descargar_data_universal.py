import ccxt
import pandas as pd
import os
import time
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
FECHA_INICIO = "2020-01-01 00:00:00" 
TIMEFRAME = '5m'

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DESTINO = os.path.join(BASE_DIR, 'DATOS_ENTRADA', 'ACTIVOS_A_OPERAR')

if not os.path.exists(DIR_DESTINO):
    os.makedirs(DIR_DESTINO)

# ==============================================================================
# LISTA MAESTRA DE ACTIVOS (100 MONEDAS TOTALES)
# ==============================================================================
# El script saltará automáticamente las que ya descargaste.

COINS_TO_DOWNLOAD = [
    # --- GRUPO 1: LA ÉLITE (Top 20) ---
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'DOT/USDT', 
    'MATIC/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT', 
    'UNI/USDT', 'XLM/USDT', 'ETC/USDT', 'FIL/USDT', 'NEAR/USDT',

    # --- GRUPO 2: EXPANSIÓN (Mid Caps & L2) ---
    'APT/USDT', 'ARB/USDT', 'OP/USDT', 'RNDR/USDT', 'INJ/USDT',
    'STX/USDT', 'IMX/USDT', 'VET/USDT', 'GRT/USDT', 'ALGO/USDT',
    'THETA/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'EGLD/USDT',
    'AAVE/USDT', 'FTM/USDT', 'EOS/USDT', 'NEO/USDT', 'XTZ/USDT',
    'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
    'LUNA/USDT', 'LUNC/USDT', 'GMT/USDT', 'GALA/USDT', 'DYDX/USDT',

    # --- GRUPO 3: LOS VETERANOS & DEFI (Nuevos 50) ---
    # Monedas con mucha historia y comportamiento único
    
    # Privacidad & Legacy (Muy cíclicas)
    'ZEC/USDT', 'DASH/USDT', 'XMR/USDT', 'IOTA/USDT', 'BAT/USDT', 
    'QTUM/USDT', 'ZIL/USDT', 'ICX/USDT', 'ONT/USDT', 'RVN/USDT',
    'IOST/USDT', 'OMG/USDT', 'ONE/USDT', 'KSM/USDT', 'ZRX/USDT',

    # DeFi 1.0 (Blue Chips - Se mueven fuerte con Ethereum)
    'MKR/USDT', 'COMP/USDT', 'SNX/USDT', 'CRV/USDT', '1INCH/USDT',
    'SUSHI/USDT', 'RUNE/USDT', 'CAKE/USDT', 'YFI/USDT', 'BAL/USDT',
    'KAVA/USDT', 'LRC/USDT', 'ENJ/USDT', 'CHZ/USDT', 'HBAR/USDT',

    # Infraestructura & Web3 (Volatilidad media/alta)
    'MINA/USDT', 'FLOW/USDT', 'KLAY/USDT', 'AR/USDT', 'LPT/USDT',
    'ENS/USDT', 'FET/USDT', 'RSR/USDT', 'OGN/USDT', 'BAKE/USDT',
    'SKL/USDT', 'C98/USDT', 'SXP/USDT', 'BLZ/USDT', 'LDO/USDT',
    'CVX/USDT', 'FXS/USDT', 'HIGH/USDT', 'PEOPLE/USDT', 'ANT/USDT'
]

# ==============================================================================
# FUNCIÓN DE DESCARGA (CON REINTENTOS)
# ==============================================================================
def descargar_historial(symbol, start_date_str):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    })
    
    since = exchange.parse8601(start_date_str)
    all_ohlcv = []
    limit = 1000 
    
    print(f"   ⬇️  Iniciando descarga: {symbol} ...", end="", flush=True)
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=limit)
            if not ohlcv: break
            
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            
            print(f"\r   ⬇️  Descargando {symbol}: {len(all_ohlcv)} velas...", end="", flush=True)
            
            if len(ohlcv) < limit: break
            if last_timestamp >= exchange.milliseconds() - 60000: break
            
        except Exception as e:
            # Si falla, esperamos un poco y seguimos (reintento simple)
            time.sleep(2)
            continue

    if not all_ohlcv: 
        print(f"\n   ❌ No se encontraron datos o par no existe.")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df[~df.index.duplicated(keep='last')]
    
    print(f" ✅ Completado.")
    return df

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    print("=========================================================")
    print(f"🚀 GESTOR DE DESCARGAS MASIVO ({len(COINS_TO_DOWNLOAD)} Activos)")
    print(f"📁 Destino: {DIR_DESTINO}")
    print("=========================================================\n")
    
    contador_nuevos = 0
    
    for symbol in COINS_TO_DOWNLOAD:
        nombre_limpio = symbol.replace('/', '') 
        filename = f"{nombre_limpio}.csv"
        path_completo = os.path.join(DIR_DESTINO, filename)
        
        # 1. Verificar existencia (SALTAR SI YA ESTÁ)
        if os.path.exists(path_completo):
            # Check rápido: si el archivo es muy chico (<1MB), quizás falló antes, lo bajamos de nuevo.
            if os.path.getsize(path_completo) > 1000: 
                print(f"⏩ {symbol}: Ya existe. Saltando.")
                continue
            
        # 2. Descargar
        try:
            df = descargar_historial(symbol, FECHA_INICIO)
            
            if df is not None and not df.empty:
                df.to_csv(path_completo, index=False)
                print(f"   💾 Guardado: {filename} ({len(df)} registros)")
                contador_nuevos += 1
            else:
                print(f"   ⚠️ {symbol}: Data vacía.")
                
        except Exception as e:
            print(f"   ❌ Error en {symbol}: {e}")

    print("\n" + "="*60)
    print(f"🎉 PROCESO FINALIZADO. Se agregaron {contador_nuevos} nuevos activos.")
    print("="*60)

if __name__ == "__main__":
    main()