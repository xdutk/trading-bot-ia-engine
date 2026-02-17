import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta  
import ccxt
import os
from sklearn.preprocessing import RobustScaler 

# --- RUTAS EXACTAS ---
PATH_MODEL = "MODELOS_ENTRENADOS/model_hmm_unsupervised.pkl"
PATH_SCALER = "DATOS_PARA_ENTRENAR_NPZ/scaler_hmm.pkl"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"

def check_hmm():
    print(f"🔬 INSPECCIONANDO CEREBRO HMM (6 FEATURES)...")
    
    # 1. Cargar Archivos
    if not os.path.exists(PATH_MODEL) or not os.path.exists(PATH_SCALER):
        print("❌ ERROR: No encuentro los archivos .pkl")
        return

    try:
        model = joblib.load(PATH_MODEL)
        scaler = joblib.load(PATH_SCALER)
        print("✅ Cerebro cargado exitosamente.")
    except Exception as e:
        print(f"❌ Error cargando pkl: {e}")
        return

    # 2. Descargar Datos (Necesitamos al menos 100 velas para calcular la SMA50 bien)
    print(f"\n🌍 BAJANDO DATOS ({SYMBOL} {TIMEFRAME})...")
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # 3. CALCULAR INDICADORES
        # ---------------------------------------------------------------------
        # A. Log Ret
        df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # B. ATR Pct
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR_Pct'] = df['ATR'] / df['close']
        
        # C. RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # D. ADX 
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx_df['ADX_14']
        
        # E. Distancia SMA 50
        sma50 = ta.sma(df['close'], length=50)
        df['Dist_SMA'] = (df['close'] - sma50) / sma50
        
        # F. Cambio Volumen
        df['Vol_Chg'] = np.log(df['volume'] / (df['volume'].shift(1) + 1))
        
        # Limpieza
        df.dropna(inplace=True)
        
        # ---------------------------------------------------------------------
        
        # 4. DATO ACTUAL (Última fila)
        last_row = df.iloc[-1]
        
        # Orden de columnas: ['Log_Ret', 'ATR_Pct', 'RSI', 'ADX', 'Dist_SMA', 'Vol_Chg']
        features_raw = np.array([
            last_row['Log_Ret'],
            last_row['ATR_Pct'],
            last_row['RSI'],
            last_row['ADX'],
            last_row['Dist_SMA'],
            last_row['Vol_Chg']
        ]).reshape(1, -1)
        
        # 5. ESCALAR Y PREDECIR
        features_scaled = scaler.transform(features_raw)
        current_state = model.predict(features_scaled)[0]
        probs = model.predict_proba(features_scaled)[0]
        
        # 6. VISUAL
        print(f"\n📊 DATOS EN TIEMPO REAL (Lo que ve la IA):")
        print(f"   1. Retorno:   {last_row['Log_Ret']*100:.2f}%")
        print(f"   2. ATR Pct:   {last_row['ATR_Pct']*100:.2f}%")
        print(f"   3. RSI (14):  {last_row['RSI']:.1f}")
        print(f"   4. ADX (14):  {last_row['ADX']:.1f}")
        print(f"   5. Dist SMA:  {last_row['Dist_SMA']*100:.2f}%")
        print(f"   6. Vol Chg:   {last_row['Vol_Chg']:.2f}")
        
        print("-" * 40)
        print(f"🤖 DIAGNÓSTICO DEL MODELO:")
        print(f"   👉 ESTADO ACTUAL: Cluster {current_state}")
        
        # Barra de confianza
        print(f"\n   🌡️  CONFIANZA (Votación interna):")
        for i, p in enumerate(probs):
            bar = "█" * int(p * 20)
            status = "👈 ELEGIDO" if i == current_state else ""
            print(f"      C{i}: {p*100:5.1f}% {bar} {status}")

    except Exception as e:
        print(f"⚠️ Error fatal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_hmm()