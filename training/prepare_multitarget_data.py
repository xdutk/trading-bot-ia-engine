import pandas as pd
import numpy as np
import glob
import os
import joblib
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_INPUT = os.path.join(BASE_DIR, 'DATOS_PROCESADOS')
DIR_OUTPUT_NPZ = os.path.join(BASE_DIR, 'DATOS_PARA_ENTRENAR_NPZ')

os.makedirs(DIR_OUTPUT_NPZ, exist_ok=True)

# PARÁMETROS GLOBALES
WINDOW_5M = 60        # 5 Horas
WINDOW_DAILY = 14     
MAX_HOLD_CANDLES = 8640 

# CONFIGURACIÓN DE ESTRATEGIAS
ESTRATEGIAS = {
    'IA_FRPV': {
        'buy': 'Real Price Buy', 'sell': 'Real Price Sell',
        'tp_mult': 14.0, 'sl_mult': 2.0   
    },
    'IA_RANGO': {
        'buy': 'Real_Price_Rango_Buy', 'sell': 'Real_Price_Rango_Sell',
        'tp_mult': 3.5, 'sl_mult': 1.5    
    },
    'IA_BREAKOUT': {
        'buy': 'Real_Price_Breakout_Buy', 'sell': 'Real_Price_Breakout_Sell',
        'tp_mult': 4.0, 'sl_mult': 1.0    
    },
    'IA_TREND': {
        'buy': 'Real_Price_Trend_Buy', 'sell': 'Real_Price_Trend_Sell',
        'tp_mult': 7.0, 'sl_mult': 2.0    
    }
}

# FEATURES
COLS_MICRO = ['f_5m_ret', 'f_5m_vol_z', 'f_dist_kama', 'f_dist_sma', 'f_dist_lrc', 'f_5m_overext']
COLS_MACRO = [
    'Retorno Diario Activo', 'Retorno Top20', 'Diferencia vs Mercado',
    'Promedio Últimos N', 'Z-Score Sesgo Ponderado', 'Correlación',
    'Z-Score Volumen', 'Vol_Signal', 'BTC_Trend_Score', 'BTC_Daily_Ret'
]

# ==============================================================================
# GENERADOR
# ==============================================================================
def generar_datasets():
    print(f"🚀 GENERANDO DATASETS MAESTROS (V4 - Fix Fechas)...")
    
    archivos = glob.glob(os.path.join(DIR_INPUT, "*.csv"))
    if not archivos: print("❌ No hay CSVs procesados."); return

    # Stores
    data_store = {k: {'X_micro': [], 'X_macro': [], 'y': []} for k in ESTRATEGIAS.keys()}
    context_store = {'X': [], 'y': []}
    tactico_store = {'X': [], 'y': []}

    total_files = len(archivos)
    
    for idx, f in enumerate(archivos):
        print(f"   [{idx+1}/{total_files}] Procesando: {os.path.basename(f)} ...")
        try:
            df = pd.read_csv(f)
            
            # --- FIX DE FECHAS ---
            # Para el resample de 1h
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], utc=True)
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
            else:
                print("   ⚠️ Archivo sin columna 'time', saltando.")
                continue

            # --- PREPARACIÓN ---
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            if 'f_5m_ret' not in df.columns: df['f_5m_ret'] = df['close'].pct_change().fillna(0)
            if 'f_5m_vol_z' not in df.columns: 
                v_mean = df['volume'].rolling(20).mean(); v_std = df['volume'].rolling(20).std()
                df['f_5m_vol_z'] = ((df['volume'] - v_mean) / (v_std + 1e-9)).fillna(0)

            # Limpieza básica
            cols_check = COLS_MICRO + COLS_MACRO + ['ATR']
            missing = [c for c in cols_check if c not in df.columns]
            if missing:
                for c in missing: df[c] = 0
            
            df.dropna(subset=cols_check, inplace=True)
            if len(df) < 500: continue

            # ==================================================================
            # 1. ESTRATEGIAS
            # ==================================================================
            for nombre_ia, params in ESTRATEGIAS.items():
                col_buy, col_sell = params['buy'], params['sell']
                if col_buy not in df.columns: continue

                indices = df[df[col_buy].notna() | df[col_sell].notna()].index
                last_exit = -1 
                
                # Convertimos indices de tiempo a posiciones enteras para el bucle
                valid_ilocs = [df.index.get_loc(x) for x in indices]

                for pos in valid_ilocs:
                    # Check overlap (pos entera)
                    if pos <= last_exit: continue
                    if pos < WINDOW_5M or pos > len(df) - MAX_HOLD_CANDLES: continue
                    
                    row = df.iloc[pos]
                    atr = row['ATR']
                    if atr <= 0 or pd.isna(atr): continue

                    is_long = not pd.isna(row[col_buy])
                    direction = 1.0 if is_long else -1.0


                    entry = row['close']
                    if is_long: tp = entry + (atr * params['tp_mult']); sl = entry - (atr * params['sl_mult'])
                    else: tp = entry - (atr * params['tp_mult']); sl = entry + (atr * params['sl_mult'])
                    
                    outcome = 0
                    limit = min(len(df), pos + MAX_HOLD_CANDLES)
                    
                    # Vectorized Lookahead
                    fut = df.iloc[pos+1 : limit]
                    if is_long:
                        hit_tp = fut[fut['high'] >= tp].index
                        hit_sl = fut[fut['low'] <= sl].index
                    else:
                        hit_tp = fut[fut['low'] <= tp].index
                        hit_sl = fut[fut['high'] >= sl].index
                    
                    # Convertimos timestamps de hit a posiciones iloc
                    first_tp = df.index.get_loc(hit_tp[0]) if len(hit_tp) > 0 else 99999999
                    first_sl = df.index.get_loc(hit_sl[0]) if len(hit_sl) > 0 else 99999999
                    
                    if first_tp < first_sl: 
                        outcome = 1; last_exit = first_tp
                    elif first_sl < first_tp: 
                        outcome = 0; last_exit = first_sl
                    else: 
                        outcome = 0; last_exit = limit

                    # Data Extraction
                    w_mic = df.iloc[pos-WINDOW_5M+1 : pos+1][COLS_MICRO].values
                    w_mac = df.iloc[pos-WINDOW_DAILY+1 : pos+1][COLS_MACRO].values
                    
                    x_mic = np.hstack((w_mic, np.full((WINDOW_5M, 1), direction)))
                    
                    data_store[nombre_ia]['X_micro'].append(x_mic)
                    data_store[nombre_ia]['X_macro'].append(w_mac)
                    data_store[nombre_ia]['y'].append(outcome)

            # ==================================================================
            # 2. CONTEXTO MACRO
            # ==================================================================
            if 'TARGET_CONTEXTO' in df.columns:
                indices = range(WINDOW_DAILY, len(df)-100, 48)
                for pos in indices:
                    w_mac = df.iloc[pos-WINDOW_DAILY+1 : pos+1][COLS_MACRO].values
                    target = df['TARGET_CONTEXTO'].iloc[pos]
                    context_store['X'].append(w_mac)
                    context_store['y'].append(target)

            # ==================================================================
            # 3. CONTEXTO TÁCTICO (1H)
            # ==================================================================
            if 'TARGET_TACTICO' in df.columns:
                df_1h = df.resample('1h').agg({
                    'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum',
                    'TARGET_TACTICO': 'last' 
                }).dropna()
                
                if len(df_1h) > 100:
                    df_1h['RSI'] = ta.rsi(df_1h['close'], length=14)
                    df_1h['ADX'] = ta.adx(df_1h['high'], df_1h['low'], df_1h['close'], length=14)['ADX_14']
                    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
                    df_1h['ATR_Norm'] = df_1h['ATR'] / df_1h['close']
                    df_1h['SMA_50'] = ta.sma(df_1h['close'], length=50)
                    df_1h['Dist_SMA'] = (df_1h['close'] - df_1h['SMA_50']) / df_1h['SMA_50']
                    df_1h['Slope'] = ta.slope(df_1h['close'], length=5)
                    df_1h.dropna(inplace=True)
                    
                    feat_cols = ['RSI', 'ADX', 'ATR_Norm', 'Dist_SMA', 'Slope']
                    vals = df_1h[feat_cols].values
                    targets = df_1h['TARGET_TACTICO'].values
                    
                    # Ventana 48 Horas
                    for i in range(48, len(df_1h)):
                        window = vals[i-48:i]
                        y_val = targets[i]
                        tactico_store['X'].append(window)
                        tactico_store['y'].append(y_val)

        except Exception as e:
            print(f"⚠️ Error en {os.path.basename(f)}: {e}")
            import traceback; traceback.print_exc()

    # ==========================================================================
    # GUARDADO FINAL
    # ==========================================================================
    print("\n💾 GUARDANDO Y ESCALANDO ...")
    
    scaler_micro = RobustScaler()
    scaler_macro = RobustScaler() 
    scaler_tact = RobustScaler()

    # A. Estrategias
    for name, data in data_store.items():
        if len(data['y']) > 0:
            X_mic = np.array(data['X_micro'])
            X_mac = np.array(data['X_macro'])
            y = np.array(data['y'])
            
            N, T, F = X_mic.shape
            feats = scaler_micro.fit_transform(X_mic[:,:,:-1].reshape(-1, F-1)).reshape(N, T, F-1)
            X_mic_final = np.concatenate([feats, X_mic[:,:,-1:]], axis=2)
            
            X_mac_final = scaler_macro.fit_transform(X_mac.reshape(-1, X_mac.shape[2])).reshape(X_mac.shape)
            
            np.savez_compressed(os.path.join(DIR_OUTPUT_NPZ, f'dataset_{name}.npz'), 
                                X_micro=X_mic_final, X_macro=X_mac_final, y=y)
            print(f"   ✅ {name}: {len(y)} muestras.")

    # B. Macro Contexto
    if len(context_store['y']) > 0:
        X = np.array(context_store['X'])
        X_s = scaler_macro.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        np.savez_compressed(os.path.join(DIR_OUTPUT_NPZ, 'dataset_IA_CONTEXTO.npz'), 
                            X=X_s, y=np.array(context_store['y']))
        print(f"   ✅ IA_CONTEXTO: {len(context_store['y'])} muestras.")

    # C. Táctico (1H)
    if len(tactico_store['y']) > 0:
        X = np.array(tactico_store['X'])
        X_s = scaler_tact.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        np.savez_compressed(os.path.join(DIR_OUTPUT_NPZ, 'dataset_IA_TACTICO_1H.npz'), 
                            X=X_s, y=np.array(tactico_store['y']))
        print(f"   ✅ IA_TACTICO: {len(tactico_store['y'])} muestras.")
        
    joblib.dump(scaler_micro, os.path.join(DIR_OUTPUT_NPZ, 'scaler_micro_global.pkl'))
    joblib.dump(scaler_macro, os.path.join(DIR_OUTPUT_NPZ, 'scaler_macro_global.pkl'))
    joblib.dump(scaler_tact, os.path.join(DIR_OUTPUT_NPZ, 'scaler_tactico_1h.pkl'))
    
    print("\n🎉 TODO LISTO PARA ENTRENAR.")

if __name__ == "__main__":
    generar_datasets()