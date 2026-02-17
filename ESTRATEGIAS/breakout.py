import pandas as pd
import pandas_ta as ta
import numpy as np

def aplicar_estrategia(df):
    """
    Estrategia de Breakout / Explosión de Volatilidad (TTM Squeeze simplificado).
    Busca momentos donde la volatilidad se comprime y luego explota.
    """
    
    # --------------------------------------------------------------------------
    # 1. BOLLINGER BANDS (20, 2)
    # --------------------------------------------------------------------------
    if 'BB_Lower' not in df.columns:
        bb = ta.bbands(df['close'], length=20, std=2)
        # pandas_ta devuelve: Lower, Mid, Upper, Bandwidth, Percent
        bb.columns = ['BB_Lower', 'BB_Mid', 'BB_Upper', 'BB_Width', 'BB_Pct']
        df = df.join(bb)
    
    # --------------------------------------------------------------------------
    # 2. KELTNER CHANNELS (20, 1.5)
    # --------------------------------------------------------------------------
    if 'KC_Lower' not in df.columns:
        kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
        kc.columns = ['KC_Lower', 'KC_Mid', 'KC_Upper']
        df = df.join(kc)
    
    # --------------------------------------------------------------------------
    # 3. VOLUMEN SMA (20)
    # --------------------------------------------------------------------------
    # Usamos 20 (Corto Plazo) porque el Breakout es un evento inmediato.
    # Queremos volumen anormal respecto a las últimas horas.
    if 'Vol_SMA' not in df.columns:
        df['Vol_SMA'] = ta.sma(df['volume'], length=20)

    # --------------------------------------------------------------------------
    # LÓGICA SQUEEZE
    # --------------------------------------------------------------------------
    # Bollinger DENTRO de Keltner = Posible volatilidad comprimida
    squeeze_on = (
        (df['BB_Lower'] > df['KC_Lower']) & 
        (df['BB_Upper'] < df['KC_Upper'])
    )
    
    # --------------------------------------------------------------------------
    # SEÑALES
    # --------------------------------------------------------------------------
    # LONG:
    # 1. Veníamos de squeeze (shift 1 es True)
    # 2. El precio rompe la Banda Superior
    # 3. Hay explosión de volumen (> 1.5 veces el promedio local)
    long_cond = (
        (squeeze_on.shift(1) == True) & 
        (df['close'] > df['BB_Upper']) & 
        (df['volume'] > df['Vol_SMA'] * 1.5)
    )
    
    # SHORT:
    # 1. Veníamos de squeeze
    # 2. El precio rompe la Banda Inferior
    # 3. Hay explosión de volumen
    short_cond = (
        (squeeze_on.shift(1) == True) & 
        (df['close'] < df['BB_Lower']) & 
        (df['volume'] > df['Vol_SMA'] * 1.5)
    )

    # --------------------------------------------------------------------------
    # SALIDAS (LABELING)
    # --------------------------------------------------------------------------
    df['Real_Price_Breakout_Buy'] = np.where(long_cond, df['close'], np.nan)
    df['Real_Price_Breakout_Sell'] = np.where(short_cond, df['close'], np.nan)
    
    return df