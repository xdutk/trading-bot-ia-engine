import pandas as pd
import numpy as np
import pandas_ta as ta

def aplicar_estrategia(df):
    """
    [ESTRATEGIA PRIVADA - VERSIÓN DEMO]
    
    NOTA PARA RECLUTADORES / NOTE TO REVIEWERS:
    La lógica propietaria exacta (KAMA/LRC/MTF Logic) ha sido ocultada 
    en este repositorio público por protección de Propiedad Intelectual.
    
    Este archivo contiene una implementación genérica de 'Mean Reversion' 
    para demostrar la estructura del código y el flujo de datos sin revelar 
    el Alpha real.
    
    ---------------------------------------------------------------------
    1. Calcula Señales de Entrada (Genéricas para Demo).
    2. Calcula Features MICRO (5m) para que el pipeline de IA no se rompa.
    """
    
    # ==============================================================================
    # A. INDICADORES TÉCNICOS
    # ==============================================================================

    df['KAMA'] = ta.ema(df['close'], length=50) 
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['LRC'] = ta.sma(df['close'], length=100) 
    
    # --------------------------------------------------------------------------
    # B. FEATURES MICRO
    # --------------------------------------------------------------------------
    epsilon = 1e-9
    
    # 1. Retorno 5m
    df['f_5m_ret'] = df['close'].pct_change()
    
    # 2. Z-Score Volumen (Estándar)
    VOL_LEN_MICRO = 200
    vol_mean = df['volume'].rolling(VOL_LEN_MICRO).mean()
    vol_std = df['volume'].rolling(VOL_LEN_MICRO).std()
    z_score_raw = (df['volume'] - vol_mean) / (vol_std + epsilon)
    df['f_5m_vol_z'] = ta.sma(z_score_raw, length=3)
    
    # 3. Distancias
    df['f_dist_kama'] = (df['close'] - df['KAMA']) / (df['KAMA'] + epsilon)
    df['f_dist_lrc']  = (df['close'] - df['LRC'])  / (df['LRC'] + epsilon)
    df['f_dist_sma']  = (df['close'] - df['SMA_200']) / (df['SMA_200'] + epsilon)
    
    # 4. Overextension
    dist_sma = df['close'] - df['SMA_200']
    mean_dist = dist_sma.abs().rolling(100).mean()
    df['f_5m_overext'] = 0
    df.loc[df['close'] > (df['SMA_200'] + mean_dist), 'f_5m_overext'] = 1  
    df.loc[df['close'] < (df['SMA_200'] - mean_dist), 'f_5m_overext'] = -1 

    # ==============================================================================
    # C. LÓGICA DE SEÑALES
    # ==============================================================================
    
    df_1h = df.resample('1h').agg({'close': 'last'}).dropna()
    df_1h['sma_1h'] = ta.sma(df_1h['close'], length=50)
    
    
    df = df.join(df_1h['sma_1h'].reindex(df.index, method='ffill'))

    # ==============================================================================
    # D. SEÑALES FINALES
    # ==============================================================================
    
    
    buy_cond = (df['close'] > df['KAMA']) & (df['f_5m_vol_z'] > 1)
    sell_cond = (df['close'] < df['KAMA']) & (df['f_5m_vol_z'] > 1)

    
    df['Real Price Buy'] = np.where(buy_cond, df['close'], np.nan)
    df['Real Price Sell'] = np.where(sell_cond, df['close'], np.nan)
    
    # Limpieza de columnas temporales
    df.drop(columns=['sma_1h'], inplace=True, errors='ignore')

    return df