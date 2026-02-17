import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import tensorflow as tf
import os
import glob
import gc
import sys
from datetime import timedelta

# ==============================================================================
# CONFIGURACIÓN (LÓGICA TOTAL: ESTRATEGIAS + HMM + MACRO + TACTICO)
# ==============================================================================
CONFIG_STRAT = {
    'FRPV':     {'tp': 14.0, 'sl': 2.0, 'buy': 'Real Price Buy', 'sell': 'Real Price Sell'},
    'RANGO':    {'tp': 3.5, 'sl': 1.5, 'buy': 'Real_Price_Rango_Buy', 'sell': 'Real_Price_Rango_Sell'},
    'BREAKOUT': {'tp': 4.0, 'sl': 1.0, 'buy': 'Real_Price_Breakout_Buy', 'sell': 'Real_Price_Breakout_Sell'},
    'TREND':    {'tp': 7.0, 'sl': 2.0, 'buy': 'Real_Price_Trend_Buy', 'sell': 'Real_Price_Trend_Sell'}
}

DIR_PROCESADOS = "DATOS_PROCESADOS"
DIR_MODELOS = "MODELOS_ENTRENADOS"
DIR_SCALERS = "DATOS_PARA_ENTRENAR_NPZ"
OUTPUT_FILE = "DATASET_GERENTE_MASIVO_V3.csv" 

FECHA_INICIO_MINERIA = "2020-01-01" 

CHUNK_SIZE = 25000   # Valores intermedios, ajustados para balancear velocidad y memoria. Se pueden aumentar si se dispone de GPU potente.
BATCH_SIZE = 4096    # Valores intermedios, ajustados para balancear velocidad y memoria. Se pueden aumentar si se dispone de GPU potente.

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Columnas requeridas
COLS_MICRO = ['f_5m_ret', 'f_5m_vol_z', 'f_dist_kama', 'f_dist_sma', 'f_dist_lrc', 'f_5m_overext']
COLS_MACRO = ['Retorno Diario Activo', 'Retorno Top20', 'Diferencia vs Mercado', 'Promedio Últimos N', 
              'Z-Score Sesgo Ponderado', 'Correlación', 'Z-Score Volumen', 'Vol_Signal', 
              'BTC_Trend_Score', 'BTC_Daily_Ret']
# Columnas Tácticas
COLS_TACTICO = ['RSI', 'ADX', 'ATR_Norm', 'Dist_SMA', 'Slope']

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    print(f"🔥 GPU Lista: {len(gpus)}")
except: pass

# ==============================================================================
# CARGA DE CEREBRO (FULL)
# ==============================================================================

def cargar_cerebro():
    print("🧠 Cargando TODO el equipo (Estrategias, HMM, Contexto y Táctica)...")
    modelos = {}
    
    # 1. Estrategias 
    for k in CONFIG_STRAT: 
        modelos[k] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, f'model_IA_{k}.keras'))
    
    # 2. HMM unsupervised (Estados de Mercado)  
    modelos['HMM'] = joblib.load(os.path.join(DIR_MODELOS, 'model_hmm_unsupervised.pkl'))
    
    # 3. CONTEXTO (MACRO) 
    modelos['CONTEXTO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_CONTEXTO.keras'))

    # 4. TACTICO (RED NEURONAL 1H)
    modelos['TACTICO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_TACTICO_1H.keras'))
    
    scalers = {}
    scalers['micro'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_micro_global.pkl'))
    scalers['macro'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_macro_global.pkl'))
    scalers['hmm'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_hmm.pkl'))
    # Scaler Táctico
    scalers['tactico'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_tactico_1h.pkl'))
    
    return modelos, scalers

# ==============================================================================
# PROCESAMIENTO
# ==============================================================================
def procesar_lote(df_chunk, modelos, scalers, symbol):
    if len(df_chunk) < 100: return []
    
    # --- A. FEATURES MACRO ---
    w_mac_raw = np.lib.stride_tricks.sliding_window_view(df_chunk[COLS_MACRO].values, window_shape=(14, len(COLS_MACRO)))
    w_mac_raw = w_mac_raw.reshape(w_mac_raw.shape[0], 14, len(COLS_MACRO))
    N_mac, T_mac, F_mac = w_mac_raw.shape
    w_mac_scaled = scalers['macro'].transform(w_mac_raw.reshape(-1, F_mac)).reshape(N_mac, T_mac, F_mac)
    
    # --- B. FEATURES MICRO ---
    w_mic_raw = np.lib.stride_tricks.sliding_window_view(df_chunk[COLS_MICRO].values, window_shape=(60, len(COLS_MICRO)))
    w_mic_raw = w_mic_raw.reshape(w_mic_raw.shape[0], 60, len(COLS_MICRO))
    N_mic, T_mic, F_mic = w_mic_raw.shape
    w_mic_scaled = scalers['micro'].transform(w_mic_raw.reshape(-1, F_mic)).reshape(N_mic, T_mic, F_mic)
    
    # --- C. PREPARACIÓN DATOS 1H (HMM + TÁCTICO) ---
    try:
        # Resampleamos el chunk a 1H
        df_1h = df_chunk.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        if len(df_1h) < 50: return [] # Necesitamos al menos 48h para la táctica

        # --- C1. Features HMM ---
        df_1h['Log_Ret'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
        df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
        df_1h['ATR_Pct'] = df_1h['ATR'] / df_1h['close']
        df_1h['RSI'] = ta.rsi(df_1h['close'], length=14)
        adx = ta.adx(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
        df_1h['ADX'] = adx['ADX_14'] if (adx is not None and 'ADX_14' in adx.columns) else 0
        sma50 = ta.sma(df_1h['close'], length=50)
        df_1h['Dist_SMA'] = (df_1h['close'] - sma50) / sma50
        df_1h['Vol_Chg'] = np.log((df_1h['volume'] + 1) / (df_1h['volume'].shift(1) + 1))
        
        # --- C2. Features TÁCTICOS (Específicos de main.py) ---
        df_1h['ATR_Norm'] = df_1h['ATR_Pct'] 
        df_1h['Slope'] = ta.slope(df_1h['close'], length=5)
        
        df_1h.dropna(inplace=True)
        
        # --- C3. Inferencia HMM ---
        X_hmm = df_1h[['Log_Ret', 'ATR_Pct', 'RSI', 'ADX', 'Dist_SMA', 'Vol_Chg']].values
        if len(X_hmm) == 0: return []
        X_hmm_s = scalers['hmm'].transform(X_hmm)
        df_1h['HMM_State'] = modelos['HMM'].predict(X_hmm_s)
        
        # --- C4. Inferencia TÁCTICA (Red Neuronal 1H) ---
        # Preparamos ventana de 48 horas
        w_tac = np.lib.stride_tricks.sliding_window_view(df_1h[COLS_TACTICO].values, window_shape=(48, len(COLS_TACTICO)))
        w_tac = w_tac.reshape(w_tac.shape[0], 48, len(COLS_TACTICO))
        
        # Scaler Tactico
        N_t, T_t, F_t = w_tac.shape
        w_tac_s = scalers['tactico'].transform(w_tac.reshape(-1, F_t)).reshape(N_t, T_t, F_t)
        
        # Predicción Masiva Tactica
        preds_tac = modelos['TACTICO'].predict(w_tac_s, batch_size=BATCH_SIZE, verbose=0)
        tac_states = np.argmax(preds_tac, axis=1) # 0=RAN, 1=BULL, 2=BEAR, 3=CAOS
        tac_probs = np.max(preds_tac, axis=1)
        
        # Asignar al DF 1H (las primeras 48h quedan NaN)
        df_1h['Tac_State'] = np.nan
        df_1h['Tac_Prob'] = np.nan
        df_1h.iloc[47:, df_1h.columns.get_loc('Tac_State')] = tac_states
        df_1h.iloc[47:, df_1h.columns.get_loc('Tac_Prob')] = tac_probs
        
        # Expandir a 5m (Forward Fill)
        # Esto hace que las velas de 14:00 a 14:55 tengan la predicción de las 14:00
        df_chunk = df_chunk.join(df_1h[['HMM_State', 'Tac_State', 'Tac_Prob']], how='left')
        df_chunk['HMM_State'] = df_chunk['HMM_State'].ffill().fillna(2).astype(int)
        df_chunk['Tac_State'] = df_chunk['Tac_State'].ffill().fillna(0).astype(int) # Default 0 (Rango) si falta
        df_chunk['Tac_Prob'] = df_chunk['Tac_Prob'].ffill().fillna(0.0)

    except Exception as e:
        # Fallback si falla el bloque 1H
        # print(f"Warning 1H: {e}") 
        df_chunk['HMM_State'] = 2 
        df_chunk['Tac_State'] = 0
        df_chunk['Tac_Prob'] = 0.0

    # ALINEACIÓN FINAL
    valid_len = min(len(w_mac_scaled), len(w_mic_scaled))
    df_sim = df_chunk.iloc[-valid_len:].copy()
    
    w_mac_in = w_mac_scaled[-valid_len:]
    w_mic_in = w_mic_scaled[-valid_len:]
    
    # --- D. INFERENCIA CONTEXTO (MACRO) ---
    preds_ctx = modelos['CONTEXTO'].predict(w_mac_in, batch_size=BATCH_SIZE, verbose=0)
    ctx_states = np.argmax(preds_ctx, axis=1)
    ctx_probs = np.max(preds_ctx, axis=1)
    
    # --- E. INFERENCIA ESTRATEGIAS ---
    dir_L = np.ones((valid_len, 60, 1), dtype=np.float32)
    dir_S = np.full((valid_len, 60, 1), -1.0, dtype=np.float32)
    preds_ia = {}
    
    X_L_mic = np.concatenate([w_mic_in, dir_L], axis=2)
    X_S_mic = np.concatenate([w_mic_in, dir_S], axis=2)
    
    for strat in CONFIG_STRAT:
        pL = modelos[strat].predict([X_L_mic, w_mac_in], batch_size=BATCH_SIZE, verbose=0).flatten()
        pS = modelos[strat].predict([X_S_mic, w_mac_in], batch_size=BATCH_SIZE, verbose=0).flatten()
        preds_ia[strat] = {'BUY': pL, 'SELL': pS}

    del w_mac_in, w_mic_in, X_L_mic, X_S_mic, dir_L, dir_S
    
    # --- F. EXTRACCIÓN DE SEÑALES ---
    experiencia = []
    # Arrays numpy para velocidad
    highs = df_sim['high'].values; lows = df_sim['low'].values
    closes = df_sim['close'].values; atrs = df_sim['ATR_14'].values
    atr_pcts = df_sim['ATR_Pct'].values; rsis = df_sim['RSI'].fillna(50).values
    
    # Estados IA (Arrays)
    hmms = df_sim['HMM_State'].values
    tacs = df_sim['Tac_State'].values
    tac_ps = df_sim['Tac_Prob'].values
    btc_trend = df_sim['BTC_Trend_Score'].fillna(0).values
    times = df_sim.index
    
    for strat, cfg in CONFIG_STRAT.items():
        for side in ['BUY', 'SELL']:
            col_sig = cfg['buy'] if side == 'BUY' else cfg['sell']
            if col_sig not in df_sim.columns: continue
            
            idxs = np.where(df_sim[col_sig].values > 0)[0]
            probs = preds_ia[strat][side]
            
            for i in idxs:
                if i >= len(closes) - 288: continue 
                prob_ia = probs[i]
                if prob_ia < 0.40: continue 

                entry = closes[i]; atr = atrs[i]
                if atr <= 0: continue
                
                if side == 'BUY': tp = entry + (atr * cfg['tp']); sl = entry - (atr * cfg['sl'])
                else: tp = entry - (atr * cfg['tp']); sl = entry + (atr * cfg['sl'])
                
                outcome = 0
                limit_lookahead = min(i+500, len(highs))
                future_h = highs[i+1 : limit_lookahead]
                future_l = lows[i+1 : limit_lookahead]
                
                if len(future_h) < 5: continue 
                hit_tp = False; hit_sl = False
                for k in range(len(future_h)):
                    h = future_h[k]; l = future_l[k]
                    if side == 'BUY':
                        if h >= tp: hit_tp = True; break
                        if l <= sl: hit_sl = True; break
                    else:
                        if l <= tp: hit_tp = True; break
                        if h >= sl: hit_sl = True; break
                
                if hit_tp and not hit_sl: outcome = 1
                elif hit_sl: outcome = 0
                else: continue
                
                experiencia.append({
                    'symbol': symbol, 'strategy': strat, 'side': side,
                    'prob_ia': prob_ia, 
                    'hmm_state': hmms[i], 
                    'ctx_state': ctx_states[i], 'ctx_prob': ctx_probs[i],
                    'tac_state': tacs[i], 'tac_prob': tac_ps[i], 
                    'atr_pct': atr_pcts[i], 'rsi': rsis[i],
                    'btc_trend': btc_trend[i], 'hour': times[i].hour,
                    'day': times[i].dayofweek, 'target': outcome
                })
                
    return experiencia

def minar_archivo(filepath, modelos, scalers):
    symbol = os.path.basename(filepath).replace('.csv', '')
    print(f"⛏️  Procesando: {symbol} ... ", end="")
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        df = df[df.index >= FECHA_INICIO_MINERIA] 
        if len(df) < 1000: print("Skip"); return []

        df[COLS_MACRO] = df[COLS_MACRO].ffill().fillna(0)
        df[COLS_MICRO] = df[COLS_MICRO].ffill().fillna(0)
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['close']
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        total_exp = []
        for i in range(0, len(df), CHUNK_SIZE):
            # Overlap mayor (300) para asegurar que la ventana táctica de 48h (576 velas 5m) tenga datos
            start_idx = max(0, i - 600) 
            end_idx = min(len(df), i + CHUNK_SIZE)
            df_chunk = df.iloc[start_idx:end_idx].copy()
            
            exp_chunk = procesar_lote(df_chunk, modelos, scalers, symbol)
            total_exp.extend(exp_chunk)
            del df_chunk, exp_chunk; gc.collect()

        print(f"✅ {len(total_exp)} registros.")
        return total_exp
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    modelos, scalers = cargar_cerebro()
    files = glob.glob(os.path.join(DIR_PROCESADOS, "*.csv"))
    print(f"\n📂 Archivos: {len(files)}")
    
    
    cols = ['symbol', 'strategy', 'side', 'prob_ia', 
            'hmm_state', 
            'ctx_state', 'ctx_prob', 
            'tac_state', 'tac_prob', 
            'atr_pct', 'rsi', 'btc_trend', 'hour', 'day', 'target']
            
    dummy = pd.DataFrame(columns=cols)
    dummy.to_csv(OUTPUT_FILE, index=False)
    
    total_filas = 0
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] ", end="")
        data = minar_archivo(f, modelos, scalers)
        if data:
            df_chunk = pd.DataFrame(data)
            df_chunk.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            total_filas += len(df_chunk)
            del df_chunk, data; gc.collect()
            
    print(f"\n🎉 MINERÍA COMPLETADA. Datos guardados en {OUTPUT_FILE}")