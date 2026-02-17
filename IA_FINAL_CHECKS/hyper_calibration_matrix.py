import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
import joblib
import pandas_ta as ta
import random

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_INPUT = os.path.join(BASE_DIR, 'DATOS_PROCESADOS')
DIR_MODELOS = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')
DIR_SCALERS = os.path.join(BASE_DIR, 'DATOS_PARA_ENTRENAR_NPZ')

PATTERN = "*.csv" 
N_FILES = 30 

# Configuración de Estrategias (Para calcular PnL)
CONFIG_STRAT = {
    'FRPV':     {'tp': 14.0, 'sl': 2.0, 'buy': 'Real Price Buy', 'sell': 'Real Price Sell'},
    'RANGO':    {'tp': 3.5, 'sl': 1.5, 'buy': 'Real_Price_Rango_Buy', 'sell': 'Real_Price_Rango_Sell'},
    'BREAKOUT': {'tp': 4.0, 'sl': 1.0, 'buy': 'Real_Price_Breakout_Buy', 'sell': 'Real_Price_Breakout_Sell'},
    'TREND':    {'tp': 7.0, 'sl': 2.0, 'buy': 'Real_Price_Trend_Buy', 'sell': 'Real_Price_Trend_Sell'}
}

COLS_MACRO = ['Retorno Diario Activo', 'Retorno Top20', 'Diferencia vs Mercado', 'Promedio Últimos N', 'Z-Score Sesgo Ponderado', 'Correlación', 'Z-Score Volumen', 'Vol_Signal', 'BTC_Trend_Score', 'BTC_Daily_Ret']

def cargar_recursos():
    print("🧠 Cargando Cerebros...")
    R = {}
    try:
        R['MACRO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_CONTEXTO.keras'))
        R['TACTICO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_TACTICO_1H.keras'))
        
        R['s_macro'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_macro_global.pkl'))
        R['s_tact'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_tactico_1h.pkl'))
    except Exception as e:
        print(f"❌ Error: {e}"); return None
    return R

def predecir_y_minar(df, R):
    # 1. MACRO (Contexto + Confianza)
    df['date_group'] = df.index.date
    df_daily = df.groupby('date_group').last().dropna(subset=COLS_MACRO)
    
    if len(df_daily) < 20: return []
    
    vals = df_daily[COLS_MACRO].values
    X_d, idx_d = [], []
    for i in range(14, len(df_daily)):
        X_d.append(vals[i-14:i])
        idx_d.append(df_daily.index[i])
        
    if not X_d: return []
    X_s = R['s_macro'].transform(np.array(X_d).reshape(-1, 10)).reshape(-1, 14, 10)
    probs = R['MACRO'].predict(X_s, verbose=0, batch_size=4096)
    
    df_res_d = pd.DataFrame({
        'Macro_Reg': np.argmax(probs, axis=1), 
        'Macro_Conf': np.max(probs, axis=1)
    }, index=pd.to_datetime(idx_d).tz_localize('UTC'))
    
    df = df.join(df_res_d.reindex(df.index, method='ffill'))

    # 2. TÁCTICO (1H + Confianza)
    df_1h = df.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    
    df_1h['RSI'] = ta.rsi(df_1h['close'], length=14)
    df_1h['ADX'] = ta.adx(df_1h['high'], df_1h['low'], df_1h['close'], length=14)['ADX_14']
    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
    df_1h['ATR_Norm'] = df_1h['ATR'] / df_1h['close']
    df_1h['SMA_50'] = ta.sma(df_1h['close'], length=50)
    df_1h['Dist_SMA'] = (df_1h['close'] - df_1h['SMA_50']) / df_1h['SMA_50']
    df_1h['Slope'] = ta.slope(df_1h['close'], length=5)
    df_1h.dropna(inplace=True)
    
    feat_tac = ['RSI', 'ADX', 'ATR_Norm', 'Dist_SMA', 'Slope']
    vals_t = df_1h[feat_tac].values
    X_t, idx_t = [], []
    for i in range(48, len(df_1h)):
        X_t.append(vals_t[i-48:i])
        idx_t.append(df_1h.index[i])
        
    if not X_t: return []
    X_st = R['s_tact'].transform(np.array(X_t).reshape(-1, 5)).reshape(-1, 48, 5)
    probs_t = R['TACTICO'].predict(X_st, verbose=0, batch_size=4096)
    
    df_res_t = pd.DataFrame({
        'Tac_Reg': np.argmax(probs_t, axis=1),
        'Tac_Conf': np.max(probs_t, axis=1)
    }, index=idx_t)
    
    df = df.join(df_res_t.reindex(df.index, method='ffill'))
    
    # 3. SIMULACIÓN DE TRADES
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    if 'Macro_Reg' not in df.columns or 'Tac_Reg' not in df.columns: return []
    
    trades = []
    cols_sig = []
    for cfg in CONFIG_STRAT.values(): cols_sig.extend([cfg['buy'], cfg['sell']])
    
    mask = df[cols_sig].notna().any(axis=1)
    indices = df.index[mask]
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['ATR'].values
    df_reset = df.reset_index()
    t_to_i = {t: i for i, t in enumerate(df_reset['time'])}
    
    for t in indices:
        idx = t_to_i[t]
        row = df.loc[t]
        
        m_reg = row['Macro_Reg']
        m_conf = row['Macro_Conf']
        t_reg = row['Tac_Reg']
        t_conf = row['Tac_Conf']
        
        if pd.isna(m_reg) or pd.isna(t_reg): continue
        
        for strat, cfg in CONFIG_STRAT.items():
            is_buy = not pd.isna(row.get(cfg['buy']))
            is_sell = not pd.isna(row.get(cfg['sell']))
            if not (is_buy or is_sell): continue
            
            side = 'BUY' if is_buy else 'SELL'
            entry = closes[idx]
            atr = atrs[idx]
            if np.isnan(atr): continue
            
            # TP Dinámico (Híbrido) para RANGO
            mtp = 3.0 if strat=='RANGO' and (m_reg==1 or m_reg==2) else cfg['tp']
            
            if side=='BUY': tp=entry+(atr*mtp); sl=entry-(atr*cfg['sl'])
            else: tp=entry-(atr*mtp); sl=entry+(atr*cfg['sl'])
            
            # Outcome rápido (500 velas)
            limit = min(len(closes), idx + 500)
            fut_h = highs[idx+1:limit]
            fut_l = lows[idx+1:limit]
            
            if len(fut_h)==0: continue
            
            if side=='BUY': hit_tp=fut_h>=tp; hit_sl=fut_l<=sl
            else: hit_tp=fut_l<=tp; hit_sl=fut_h>=sl
                
            itp = np.argmax(hit_tp) if hit_tp.any() else 99999
            isl = np.argmax(hit_sl) if hit_sl.any() else 99999
            
            if itp < isl: out=1
            elif isl < itp: out=-1
            else: out=0
            
            trades.append({
                'Strat': strat,
                'Macro_Reg': int(m_reg),
                'Macro_Conf': m_conf,
                'Tac_Reg': int(t_reg),
                'Tac_Conf': t_conf,
                'Outcome': out
            })
            
    return trades

def analizar_calibracion(all_trades):
    df = pd.DataFrame(all_trades)
    if df.empty: print("No hay trades."); return
    
    labels = ['RANGO', 'BULL', 'BEAR', 'CAOS']
    
    print("\n" + "="*80)
    print("🎯 CALIBRACIÓN MACRO: ¿CUÁNTO CONFIAR EN EL JEFE?")
    print("="*80)
    
    # Para cada régimen Macro, probamos umbrales de confianza
    for reg in range(4):
        label = labels[reg]
        print(f"\n--- MACRO: {label} ---")
        
        sub_df = df[df['Macro_Reg'] == reg]
        if sub_df.empty: print(" (Sin datos)"); continue
        
        print(f"   {'Confianza >':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Calidad'}")
        
        for th in [0.50, 0.60, 0.70, 0.80, 0.90]:
            mask = sub_df['Macro_Conf'] >= th
            n = mask.sum()
            if n > 50:
                wins = (sub_df[mask]['Outcome'] == 1).sum()
                wr = (wins / n) * 100
                
                # Calidad subjetiva (WR > 35% es bueno con TPs altos)
                calidad = "✅" if wr > 35 else "⚠️" if wr > 30 else "❌"
                print(f"   {th:.2f}         | {n:<8} | {wr:.2f}%     | {calidad}")
            else:
                print(f"   {th:.2f}         | {n:<8} | -          | (Pocos datos)")

    print("\n" + "="*80)
    print("🎯 CALIBRACIÓN TÁCTICA: ¿CUÁNTO CONFIAR EN EL DT (1H)?")
    print("="*80)
    
    for reg in range(4):
        label = labels[reg]
        print(f"\n--- TÁCTICO: {label} ---")
        
        sub_df = df[df['Tac_Reg'] == reg]
        if sub_df.empty: print(" (Sin datos)"); continue
        
        print(f"   {'Confianza >':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Calidad'}")
        
        # Umbrales más bajos para Táctico porque es más específico
        for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
            mask = sub_df['Tac_Conf'] >= th
            n = mask.sum()
            if n > 50:
                wins = (sub_df[mask]['Outcome'] == 1).sum()
                wr = (wins / n) * 100
                calidad = "✅" if wr > 35 else "⚠️" if wr > 30 else "❌"
                print(f"   {th:.2f}         | {n:<8} | {wr:.2f}%     | {calidad}")
            else:
                print(f"   {th:.2f}         | {n:<8} | -          | (Pocos datos)")

def main():
    R = cargar_recursos()
    if R is None: return
    
    files = glob.glob(os.path.join(DIR_INPUT, PATTERN))
    if len(files) > N_FILES:
        files = random.sample(files, N_FILES)
        
    print(f"🚀 Escaneando {len(files)} activos...")
    all_trades = []
    
    for f in files:
        try:
            print(f"   > {os.path.basename(f)}", end="\r")
            df = pd.read_csv(f)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], utc=True)
                df.set_index('time', inplace=True)
            
            trades = predecir_y_minar(df, R)
            all_trades.extend(trades)
        except: pass
        
    print(f"\n✅ {len(all_trades)} trades simulados.")
    analizar_calibracion(all_trades)

if __name__ == "__main__":
    main()