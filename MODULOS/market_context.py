import pandas as pd
import numpy as np
import pandas_ta as ta
import glob
import os

# ==============================================================================
# 1. CONSTRUCTOR DEL INDICE (BASKET RETURN) + BTC
# ==============================================================================
def construir_indice_mercado(carpeta_top20):
    print(f"🏗️  Construyendo Índice de Mercado (Top 20) y Extrayendo BTC...")
    archivos = glob.glob(os.path.join(carpeta_top20, "*.csv"))
    if not archivos: return None, None

    dict_returns = {}
    dict_volumes = {}
    btc_series = None

    for f in archivos:
        try:
            df = pd.read_csv(f, usecols=['time', 'close', 'volume'])
            # Normalizar tiempo
            if df['time'].dtype == 'int64' or df['time'].dtype == 'float64':
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            else:
                df['time'] = pd.to_datetime(df['time'], utc=True)
            
            df.set_index('time', inplace=True)
            df = df[~df.index.duplicated(keep='last')].sort_index()

            ret = df['close'].pct_change()
            name = os.path.basename(f)
            
            dict_returns[name] = ret
            dict_volumes[name] = df['volume']
            
            # --- BTC ---
            # Buscamos si el archivo es de Bitcoin para guardar su serie aparte
            if 'BTC' in name.upper():
                btc_series = df['close'].copy()
                print("   👑 Bitcoin detectado y aislado.")
                
        except: pass

    # Construir Índice Ponderado
    df_rets = pd.DataFrame(dict_returns).ffill(limit=5)
    df_vols = pd.DataFrame(dict_volumes).ffill(limit=5).fillna(0)

    numerator = (df_rets * df_vols).sum(axis=1)
    denominator = df_vols.sum(axis=1)
    market_index = numerator / (denominator + 1e-9)
    
    # Retornamos AMBOS: el índice general y la serie de BTC
    return market_index, btc_series

# ==============================================================================
# 2. INDICADORES DE CONTEXTO (RAW / PURO / DIARIO)
# ==============================================================================
def agregar_indicadores_contexto(df, market_series_5m, btc_series_5m=None):
    
    # --- RESAMPLE DIARIO (Cierre 00:00 UTC) ---
    df_daily = df.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    
    # Indice Mercado Diario
    market_price_index = (1 + market_series_5m).cumprod()
    market_daily_close = market_price_index.resample('D').last()
    
    # BTC Diario (Si existe)
    if btc_series_5m is not None:
        btc_daily_close = btc_series_5m.resample('D').last()
    else:
        # Si no hay BTC, llenamos con ceros para no romper el código
        btc_daily_close = pd.Series(0, index=df_daily.index)
    
    # Alinear todos los índices
    common_index = df_daily.index.intersection(market_daily_close.index).intersection(btc_daily_close.index)
    df_daily = df_daily.loc[common_index]
    market_daily_close = market_daily_close.loc[common_index]
    btc_daily_close = btc_daily_close.loc[common_index]

    # --- PARÁMETROS ---
    SMOOTH_LEN = 5
    CORREL_WINDOW = 14
    LOOKBACK = 50
    BIAS_LEN = 50           
    BIAS_WEIGHT_START = 20 

    # 1. Retornos
    asset_ret_d = df_daily['close'].pct_change()
    market_ret_d = market_daily_close.pct_change()
    
    # 2. Suavizado
    asset_ret_smooth = ta.sma(asset_ret_d, length=SMOOTH_LEN)
    market_ret_smooth = ta.sma(market_ret_d, length=SMOOTH_LEN)
    
    df_daily['Retorno Diario Activo'] = asset_ret_smooth
    df_daily['Retorno Top20'] = market_ret_smooth

    # 3. Diferencia y Sesgo
    perf_diff = asset_ret_smooth - market_ret_smooth
    df_daily['Diferencia vs Mercado'] = perf_diff
    
    avg_perf_diff = ta.sma(perf_diff, length=LOOKBACK)
    df_daily['Promedio Últimos N'] = avg_perf_diff
    
    # Sesgo Ponderado
    weights = np.concatenate([np.ones(BIAS_LEN - BIAS_WEIGHT_START), np.full(BIAS_WEIGHT_START, 2.0)])
    
    def weighted_zscore_calc(x):
        if len(x) != BIAS_LEN: return np.nan
        w_mean = np.average(x, weights=weights)
        variance = np.average((x - w_mean)**2, weights=weights)
        w_std = np.sqrt(variance)
        if w_std < 1e-12: w_std = 1e-12
        return (x[-1] - w_mean) / w_std

    # VALOR PURO
    df_daily['Z-Score Sesgo Ponderado'] = avg_perf_diff.rolling(window=BIAS_LEN).apply(weighted_zscore_calc, raw=True)
    
    # 4. Correlación
    df_daily['Correlación'] = asset_ret_smooth.rolling(CORREL_WINDOW).corr(market_ret_smooth)
    
    # 5. Volumen Diario (Z-Score)
    vol_len = 200 
    vol_ma = ta.sma(df_daily['volume'], length=vol_len)
    vol_std = ta.stdev(df_daily['volume'], length=vol_len)
    
    z_vol_raw = (df_daily['volume'] - vol_ma) / (vol_std + 1e-9)
    df_daily['Z-Score Volumen'] = ta.sma(z_vol_raw, length=3)
    df_daily['Vol_Signal'] = ta.sma(df_daily['Z-Score Volumen'], length=15)

    # --- INDICADORES DE BITCOIN ---
    # Calculamos la tendencia de BTC (Distancia a SMA 20 diaria)
    # "¿BTC está alcista o bajista hoy?"
    btc_sma50 = ta.sma(btc_daily_close, length=50)
    df_daily['BTC_Trend_Score'] = (btc_daily_close - btc_sma50) / (btc_sma50 + 1e-9)
    
    # Retorno diario de BTC suavizado (Momentum)
    btc_ret = btc_daily_close.pct_change()
    df_daily['BTC_Daily_Ret'] = ta.sma(btc_ret, length=3)

    # --- SHIFT DE SEGURIDAD (ANTIMIRADA AL FUTURO) ---
    cols_to_merge = [
        'Retorno Diario Activo', 'Retorno Top20',
        'Diferencia vs Mercado', 'Promedio Últimos N', 
        'Z-Score Sesgo Ponderado', 'Correlación', 
        'Z-Score Volumen', 'Vol_Signal',
        'BTC_Trend_Score', 'BTC_Daily_Ret'
    ]
    
    df_context_shifted = df_daily[cols_to_merge].shift(1)
    
    # --- MERGE ---
    df_context_final = df_context_shifted.reindex(df.index, method='ffill')
    df = df.join(df_context_final)
    
    return df