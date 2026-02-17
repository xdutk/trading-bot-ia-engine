import pandas as pd
import pandas_ta as ta
import numpy as np

def aplicar_estrategia(df):
    """
    Estrategia de Rango / Reversión a la Media.
    Compra en soportes dinámicos (Bollinger) cuando el mercado está "tranquilo" (ADX bajo).
    """
    
    # --------------------------------------------------------------------------
    # 1. INDICADORES (Con validación de existencia)
    # --------------------------------------------------------------------------
    
    # Bandas de Bollinger (20, 2)
    if 'BB_Lower' not in df.columns:
        bb = ta.bbands(df['close'], length=20, std=2)
        
        bb.columns = ['BB_Lower', 'BB_Mid', 'BB_Upper', 'BB_Width', 'BB_Pct']
        df = df.join(bb)
    
    # RSI (14)
    if 'RSI' not in df.columns:
        df['RSI'] = ta.rsi(df['close'], length=14)
    
    # ADX (14) - Filtro de Tendencia
    if 'ADX' not in df.columns:
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        # ADX devuelve 3 columnas (ADX, DMP, DMN)
        df['ADX'] = adx.iloc[:, 0]

    # --------------------------------------------------------------------------
    # 2. REGLAS DE ENTRADA
    # --------------------------------------------------------------------------
    
    # LONG:
    # 1. Precio cierra por debajo o tocando la Banda Inferior (Sobreventa estadística)
    # 2. RSI < 35 (Sobreventa de momentum)
    # 3. ADX < 25 (Ausencia de tendencia fuerte -> El precio tiende a volver al centro)
    long_cond = (
        (df['close'] <= df['BB_Lower']) & 
        (df['RSI'] < 35) & 
        (df['ADX'] < 25)
    )
    
    # SHORT:
    # 1. Precio cierra por encima o tocando la Banda Superior
    # 2. RSI > 65
    # 3. ADX < 25
    short_cond = (
        (df['close'] >= df['BB_Upper']) & 
        (df['RSI'] > 65) & 
        (df['ADX'] < 25)
    )
    
    # --------------------------------------------------------------------------
    # 3. SALIDAS (LABELING)
    # --------------------------------------------------------------------------
    df['Real_Price_Rango_Buy'] = np.where(long_cond, df['close'], np.nan)
    df['Real_Price_Rango_Sell'] = np.where(short_cond, df['close'], np.nan)
    
    return df