import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
from datetime import timedelta, timezone
import gc
import sys
import warnings

# Silenciar advertencias
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ==============================================================================
# 1. CONFIGURACIÓN IDÉNTICA AL BOT
# ==============================================================================
CAPITAL_INICIAL = 1000.0
CAPITAL_MAXIMO_OP = 50.0 
COSTO_COMISION = 0.0005 
LEVERAGE_BASE = 3

# INTENTO DE GPU
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU NVIDIA DETECTADA: {len(gpus)} dispositivo(s).")
    else:
        print("⚠️ GPU NO DETECTADA. Usando CPU.")
except: pass

# FILTROS
MAX_TRADES_GLOBAL = 8
MAX_TRADES_STRAT = 3
COOLDOWN_CANDLES = 25       
COOLDOWN_GLOBAL_CANDLES = 30 
MAX_FAILURES_STRAT = 3      

UMBRAL_CONTEXTO = 0.54
UMBRALES_TACTICOS = {0: 0.34, 1: 0.30, 2: 0.32, 3: 0.38}

REGLAS_HMM = {
    0: ['RANGO_SHORT', 'BREAKOUT_SHORT', 'TREND_SHORT', 'FRPV_SHORT'], 
    1: ['RANGO', 'BREAKOUT_LONG', 'TREND_LONG', 'FRPV_LONG'], 
    2: ['RANGO', 'BREAKOUT_SHORT', 'FRPV'],
    3: [] 
}

# --- UMBRALES REALES ---
CONFIG_STRAT = {
    'FRPV':     {'tp': 14.0, 'sl': 2.0, 'umbral': 0.50, 'buy': 'Real Price Buy', 'sell': 'Real Price Sell'},
    'RANGO':    {'tp': 3.5, 'sl': 1.5, 'umbral': 0.58, 'buy': 'Real_Price_Rango_Buy', 'sell': 'Real_Price_Rango_Sell'},
    'BREAKOUT': {'tp': 4.0, 'sl': 1.0, 'umbral': 0.65, 'buy': 'Real_Price_Breakout_Buy', 'sell': 'Real_Price_Breakout_Sell'},
    'TREND':    {'tp': 7.0, 'sl': 2.0, 'umbral': 0.63, 'buy': 'Real_Price_Trend_Buy', 'sell': 'Real_Price_Trend_Sell'}
}

# RUTAS
DIR_PROCESADOS = "DATOS_PROCESADOS"
DIR_MODELOS = "MODELOS_ENTRENADOS"
DIR_SCALERS = "DATOS_PARA_ENTRENAR_NPZ"

ACTIVOS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
    'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'TONUSDT',
    'DOTUSDT', 'LINKUSDT', 'BCHUSDT', 'LTCUSDT', 'UNIUSDT',
    'NEARUSDT', 'POLUSDT', 'APTUSDT', 'SUIUSDT', 'XLMUSDT',
    'DASHUSDT', 'ZECUSDT'
]

COLS_MICRO = ['f_5m_ret', 'f_5m_vol_z', 'f_dist_kama', 'f_dist_sma', 'f_dist_lrc', 'f_5m_overext']
COLS_MACRO = ['Retorno Diario Activo', 'Retorno Top20', 'Diferencia vs Mercado', 'Promedio Últimos N', 
              'Z-Score Sesgo Ponderado', 'Correlación', 'Z-Score Volumen', 'Vol_Signal', 
              'BTC_Trend_Score', 'BTC_Daily_Ret']

# ==============================================================================
# 2. ESTADO GLOBAL
# ==============================================================================
class GlobalSimState:
    def __init__(self):
        self.full_log = [] 

def calcular_volatility_scaling(atr_pct):
    if atr_pct <= 0: return 1.0
    factor = np.exp(-atr_pct * 55.0)
    return max(0.40, factor)

def obtener_umbrales_ajustados_local(historial_local, lado):
    ctx = UMBRAL_CONTEXTO
    tac = UMBRALES_TACTICOS.copy()
    if len(historial_local) < 10: return ctx, tac
    
    recent = historial_local[-30:]
    wins_l = len([x for x in recent if x['lado']=='BUY' and x['res']=='WIN'])
    total_l = len([x for x in recent if x['lado']=='BUY'])
    wr_l = wins_l/total_l if total_l > 0 else 0.5
    
    wins_s = len([x for x in recent if x['lado']=='SELL' and x['res']=='WIN'])
    total_s = len([x for x in recent if x['lado']=='SELL'])
    wr_s = wins_s/total_s if total_s > 0 else 0.5
    
    delta = wr_l - wr_s
    ajuste = -(delta * 0.12) if (lado=='BUY' and delta>0) or (lado=='SELL' and delta<0) else abs(delta)*0.02
    
    ctx = max(0.45, min(ctx + ajuste, 0.60))
    for k in tac: tac[k] = max(0.23, min(tac[k] + (ajuste*0.5), 0.40))
    return ctx, tac

# ==============================================================================
# 3. MOTOR IA
# ==============================================================================
def cargar_cerebro():
    print("🧠 Cargando Modelos IA...")
    try:
        modelos = {}
        for k in CONFIG_STRAT: modelos[k] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, f'model_IA_{k}.keras'))
        modelos['CONTEXTO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_CONTEXTO.keras'))
        modelos['TACTICO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_TACTICO_1H.keras'))
        modelos['HMM'] = joblib.load(os.path.join(DIR_MODELOS, 'model_hmm_unsupervised.pkl'))
        
        scalers = {}
        scalers['micro'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_micro_global.pkl'))
        scalers['macro'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_macro_global.pkl'))
        scalers['hmm'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_hmm.pkl'))
        
        print("✅ Modelos cargados correctamente.")
        return modelos, scalers
    except Exception as e:
        print(f"❌ Error fatal cargando modelos: {e}")
        sys.exit()

def procesar_activo(symbol, modelos, scalers, global_state):
    path = os.path.join(DIR_PROCESADOS, f"{symbol}.csv")
    if not os.path.exists(path): return
    
    print(f"  🔹 Analizando {symbol} ... ", end="")
    try:
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        start_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=730)
        df = df[df.index >= start_date]
        if len(df) < 1000: 
            print("Datos insuficientes.")
            return

        # A. PREPARACIÓN TENSORES
        df[COLS_MACRO] = df[COLS_MACRO].ffill().fillna(0)
        w_mac_raw = np.lib.stride_tricks.sliding_window_view(df[COLS_MACRO].values, window_shape=(14, len(COLS_MACRO)))
        w_mac_raw = w_mac_raw.reshape(w_mac_raw.shape[0], 14, len(COLS_MACRO))
        
        N, T, F = w_mac_raw.shape
        w_mac_scaled = scalers['macro'].transform(w_mac_raw.reshape(-1, F)).reshape(N, T, F)
        
        df[COLS_MICRO] = df[COLS_MICRO].ffill().fillna(0)
        w_mic_raw = np.lib.stride_tricks.sliding_window_view(df[COLS_MICRO].values, window_shape=(60, len(COLS_MICRO)))
        w_mic_raw = w_mic_raw.reshape(w_mic_raw.shape[0], 60, len(COLS_MICRO))
        
        N2, T2, F2 = w_mic_raw.shape
        w_mic_scaled = scalers['micro'].transform(w_mic_raw.reshape(-1, F2)).reshape(N2, T2, F2)
        
        valid_len = min(len(w_mac_scaled), len(w_mic_scaled))
        w_mac_in = w_mac_scaled[-valid_len:]
        w_mic_in = w_mic_scaled[-valid_len:]
        
        # B. HMM & Features 1H
        df_1h = df.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        df_1h['Log_Ret'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
        df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
        df_1h['ATR_Pct'] = df_1h['ATR'] / df_1h['close']
        df_1h['RSI'] = ta.rsi(df_1h['close'], length=14)
        adx = ta.adx(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
        df_1h['ADX'] = adx['ADX_14'] if (adx is not None and 'ADX_14' in adx.columns) else 0
        sma50 = ta.sma(df_1h['close'], length=50)
        df_1h['Dist_SMA'] = (df_1h['close'] - sma50) / sma50
        df_1h['Vol_Chg'] = np.log((df_1h['volume'] + 1) / (df_1h['volume'].shift(1) + 1))
        
        df_1h = df_1h.replace([np.inf, -np.inf], np.nan).dropna()
        if df_1h.empty: return

        X_hmm = df_1h[['Log_Ret', 'ATR_Pct', 'RSI', 'ADX', 'Dist_SMA', 'Vol_Chg']].values
        X_hmm_s = scalers['hmm'].transform(X_hmm)
        df_1h['HMM_State'] = modelos['HMM'].predict(X_hmm_s)
        
        df['HMM_State'] = df_1h['HMM_State'].reindex(df.index, method='ffill').fillna(2)
        df['Vol_Trigger'] = (df_1h['ATR_Pct'] > 0.20) | (np.abs(df_1h['Log_Ret']) > 0.20)
        vol_trigger_series = df['Vol_Trigger'].reindex(df.index, method='ffill').fillna(False)

        # C. INFERENCIA MASIVA
        dir_L = np.ones((valid_len, 60, 1))
        dir_S = np.full((valid_len, 60, 1), -1.0)
        
        X_L = [np.concatenate([w_mic_in, dir_L], axis=2), w_mac_in]
        X_S = [np.concatenate([w_mic_in, dir_S], axis=2), w_mac_in]
        
        preds = {}
        BS = 1024 
        for strat in CONFIG_STRAT:
            pL = modelos[strat].predict(X_L, batch_size=BS, verbose=0).flatten()
            pS = modelos[strat].predict(X_S, batch_size=BS, verbose=0).flatten()
            preds[strat] = {'L': pL, 'S': pS}

        # D. SIMULACIÓN LOCAL
        df_sim = df.iloc[-valid_len:].copy()
        
        times = df_sim.index
        closes = df_sim['close'].values
        highs = df_sim['high'].values
        lows = df_sim['low'].values
        
        df_sim['ATR_14'] = ta.atr(df_sim['high'], df_sim['low'], df_sim['close'], length=14)
        atrs = df_sim['ATR_14'].fillna(0).values
        atr_pcts = (atrs / closes)
        
        hmm_vals = df_sim['HMM_State'].values.astype(int)
        vol_vals = vol_trigger_series.iloc[-valid_len:].values
        
        signals = {}
        for strat, cfg in CONFIG_STRAT.items():
            for s in ['buy', 'sell']:
                col = cfg[s]
                if col not in df_sim.columns:
                    alt = col.replace('Trend', 'TREND')
                    if alt in df_sim.columns: col = alt
                
                if col in df_sim.columns:
                    signals[f"{strat}_{s}"] = df_sim[col].values
                else:
                    signals[f"{strat}_{s}"] = np.zeros(len(df_sim))

        active_trades = {} 
        local_history = []
        strat_fails = {k:0 for k in CONFIG_STRAT}
        
        MIN_TIME_UTC = pd.Timestamp("2000-01-01", tz='UTC')
        strat_cooldown = {k: MIN_TIME_UTC for k in CONFIG_STRAT}
        pair_cooldown = {k: MIN_TIME_UTC for k in CONFIG_STRAT}
        vol_block_until = MIN_TIME_UTC
        
        trades_count = 0
        
        for i in range(len(df_sim)):
            t = times[i]
            
            # 1. Salidas
            to_remove = []
            for tid, tr in active_trades.items():
                dist = abs(tr['tp'] - tr['entry'])
                if not tr['be']:
                    if tr['side']=='BUY' and (highs[i] - tr['entry']) >= dist*0.8:
                        tr['sl'] = tr['entry'] * 1.0015; tr['be'] = True
                    elif tr['side']=='SELL' and (tr['entry'] - lows[i]) >= dist*0.8:
                        tr['sl'] = tr['entry'] * 0.9985; tr['be'] = True
                
                hit_tp, hit_sl = False, False
                exit_p = 0
                if tr['side']=='BUY':
                    if highs[i] >= tr['tp']: hit_tp=True; exit_p=tr['tp']
                    elif lows[i] <= tr['sl']: hit_sl=True; exit_p=tr['sl']
                else:
                    if lows[i] <= tr['tp']: hit_tp=True; exit_p=tr['tp']
                    elif highs[i] >= tr['sl']: hit_sl=True; exit_p=tr['sl']
                
                if hit_tp or hit_sl:
                    res = 'WIN' if hit_tp else 'LOSS'
                    pct = (exit_p - tr['entry']) / tr['entry']
                    if tr['side']=='SELL': pct *= -1
                    gross = tr['margin'] * pct * tr['lev']
                    fees = (tr['margin'] * tr['lev']) * COSTO_COMISION * 2
                    net = gross - fees
                    
                    log_entry = {
                        'symbol': symbol, 'strat': tr['strat'], 'side': tr['side'],
                        'entry_time': tr['time'], 'exit_time': t,
                        'pnl_usd': net, 'res': res, 'margin': tr['margin']
                    }
                    global_state.full_log.append(log_entry)
                    local_history.append({'lado':tr['side'], 'res':res})
                    
                    if res == 'WIN': strat_fails[tr['strat']] = 0
                    else: 
                        pair_cooldown[tr['strat']] = t + timedelta(minutes=5*COOLDOWN_CANDLES)
                        strat_fails[tr['strat']] += 1
                        if strat_fails[tr['strat']] >= MAX_FAILURES_STRAT:
                            strat_cooldown[tr['strat']] = t + timedelta(minutes=5*COOLDOWN_GLOBAL_CANDLES)
                            strat_fails[tr['strat']] = 0
                    to_remove.append(tid)
            for tid in to_remove: del active_trades[tid]

            # 2. Entradas
            if vol_vals[i]: vol_block_until = t + timedelta(hours=4)
            if t < vol_block_until: continue
            if atr_pcts[i] > 0.20: continue
            if len(active_trades) >= 1: continue 
            
            hmm = int(hmm_vals[i])
            reglas = REGLAS_HMM.get(hmm, [])
            ctx_L, tac_L = obtener_umbrales_ajustados_local(local_history, 'BUY')
            ctx_S, tac_S = obtener_umbrales_ajustados_local(local_history, 'SELL')
            
            for strat, cfg in CONFIG_STRAT.items():
                if strat_fails[strat] >= MAX_FAILURES_STRAT: continue
                if t < strat_cooldown[strat]: continue
                if t < pair_cooldown[strat]: continue
                
                s_b = signals[f"{strat}_buy"][i] > 0
                s_s = signals[f"{strat}_sell"][i] > 0
                if not s_b and not s_s: continue
                
                probL = preds[strat]['L'][i]
                probS = preds[strat]['S'][i]
                side = None
                
                
                min_prob = cfg['umbral'] 
                
                if s_b:
                    permiso = (strat in reglas) or (f"{strat}_LONG" in reglas)
                    if permiso and probL >= min_prob: side = 'BUY'
                
                if s_s and side is None:
                    permiso = (strat in reglas) or (f"{strat}_SHORT" in reglas)
                    if permiso and probS >= min_prob: side = 'SELL'
                
                if side:
                    atr = atrs[i]
                    if atr <= 0: continue
                    ep = closes[i]
                    cap = CAPITAL_MAXIMO_OP * calcular_volatility_scaling(atr_pcts[i])
                    
                    if side=='BUY': tp=ep+(atr*cfg['tp']); sl=ep-(atr*cfg['sl'])
                    else: tp=ep-(atr*cfg['tp']); sl=ep+(atr*cfg['sl'])
                    
                    tid = f"{i}"
                    active_trades[tid] = {
                        'symbol': symbol, 'strat': strat, 'side': side,
                        'entry': ep, 'tp': tp, 'sl': sl, 'lev': LEVERAGE_BASE,
                        'margin': cap, 'be': False, 'time': t
                    }
                    trades_count += 1
                    break 
                    
        print(f"✅ {trades_count} trades generados.")
        del w_mac_raw, w_mic_raw, w_mac_scaled, w_mic_scaled, X_L, X_S, preds
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error en {symbol}: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 4. REPORTE FINAL
# ==============================================================================
def generar_reporte_detallado(full_log):
    if not full_log:
        print("⚠️ No hay trades para reportar.")
        return

    df = pd.DataFrame(full_log)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.sort_values('exit_time')
    
    df['cum_pnl'] = df['pnl_usd'].cumsum() + CAPITAL_INICIAL
    
    total_trades = len(df)
    wins = len(df[df['pnl_usd'] > 0])
    win_rate = (wins / total_trades) * 100
    total_pnl = df['pnl_usd'].sum()
    profit_factor = df[df['pnl_usd']>0]['pnl_usd'].sum() / abs(df[df['pnl_usd']<0]['pnl_usd'].sum()) if len(df[df['pnl_usd']<0]) > 0 else 999
    
    print("\n" + "="*60)
    print("📊 REPORTE DE INTELIGENCIA ARTIFICIAL (BACKTEST V14)")
    print("="*60)
    print(f"💰 Balance Final:   ${CAPITAL_INICIAL + total_pnl:.2f}")
    print(f"📈 Retorno Total:   {total_pnl:.2f} USDT ({(total_pnl/CAPITAL_INICIAL)*100:.2f}%)")
    print(f"🎲 Total Trades:    {total_trades}")
    print(f"🎯 Win Rate:        {win_rate:.2f}%")
    print(f"⚖️ Profit Factor:   {profit_factor:.2f}")
    
    print("\n🏆 RANKING DE ESTRATEGIAS (PnL):")
    print(df.groupby('strat')['pnl_usd'].sum().sort_values(ascending=False))
    
    print("\n📉 PEORES ACTIVOS:")
    print(df.groupby('symbol')['pnl_usd'].sum().sort_values().head(5))

    plt.figure(figsize=(12, 6))
    plt.plot(df['exit_time'], df['cum_pnl'], label='Equity Curve', color='lime')
    plt.axhline(y=CAPITAL_INICIAL, color='r', linestyle='--')
    plt.title(f"Equity Curve - {win_rate:.1f}% WinRate")
    plt.grid(True, alpha=0.3)
    # --- PARA WSL ---
    nombre_imagen = "resultado_backtest_v14.png"
    plt.savefig(nombre_imagen)
    print(f"\n📸 Gráfico guardado exitosamente como: {nombre_imagen}")
    print("   (Búscalo en tu carpeta de Windows)")
    # plt.show()

if __name__ == "__main__":
    modelos, scalers = cargar_cerebro()
    state = GlobalSimState()
    
    print("\n🚀 INICIANDO BACKTEST MASIVO (BATCH PROCESSING)...")
    for activo in ACTIVOS:
        procesar_activo(activo, modelos, scalers, state)
        
    generar_reporte_detallado(state.full_log)