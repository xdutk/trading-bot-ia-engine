import pandas as pd
import pandas_ta as ta
import numpy as np

def aplicar_estrategia(df):
    """
    Estrategia Trend Following Clásica (Golden Cross / Death Cross).
    Cruces de EMA 50 y EMA 200.
    """
    
    # --------------------------------------------------------------------------
    # 1. INDICADORES
    # --------------------------------------------------------------------------
    
    if 'EMA_50' not in df.columns:
        df['EMA_50'] = ta.ema(df['close'], length=50)
        
    if 'EMA_200' not in df.columns: 
        
        df['EMA_200'] = ta.ema(df['close'], length=200)

    # --------------------------------------------------------------------------
    # 2. SEÑALES (Lógica Vectorizada Manual)
    # --------------------------------------------------------------------------
    
    
    # LONG (Golden Cross): 
    # Hoy EMA50 está ARRIBA de EMA200 Y Ayer estaba ABAJO o IGUAL
    long_cond = (
        (df['EMA_50'] > df['EMA_200']) & 
        (df['EMA_50'].shift(1) <= df['EMA_200'].shift(1))
    )
    
    # SHORT (Death Cross):
    # Hoy EMA50 está ABAJO de EMA200 Y Ayer estaba ARRIBA o IGUAL
    short_cond = (
        (df['EMA_50'] < df['EMA_200']) & 
        (df['EMA_50'].shift(1) >= df['EMA_200'].shift(1))
    )

    # --------------------------------------------------------------------------
    # 3. SALIDAS (LABELING)
    # --------------------------------------------------------------------------
    df['Real_Price_Trend_Buy'] = np.where(long_cond, df['close'], np.nan)
    df['Real_Price_Trend_Sell'] = np.where(short_cond, df['close'], np.nan)
    
    return df