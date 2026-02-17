import pandas as pd
import numpy as np
import pandas_ta as ta

def etiquetar_regimen_mercado(df):
    """
    Etiquetado de Régimen de Mercado (Contexto) mirando el FUTURO.
    Usa UMBRALES DINÁMICOS basados en la volatilidad del propio activo.
    
    Labels:
    0 = RANGO / CALMA (Sideways)
    1 = TENDENCIA ALCISTA (Bull)
    2 = TENDENCIA BAJISTA (Bear)
    3 = ALTA VOLATILIDAD / CAOS (Noise)
    """
    
    # --------------------------------------------------------------------------
    # 0. RESAMPLE A 4H
    # --------------------------------------------------------------------------
    df_macro = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    # --------------------------------------------------------------------------
    # 1. CONFIGURACIÓN DE TIEMPOS
    # --------------------------------------------------------------------------
    # Horizonte de Predicción: 5 Días
    LOOK_AHEAD = 30  # 30 velas de 4h = 120h = 5 días
    
    # Horizonte de Referencia: 15 Días
    # Usamos más historia para definir qué es "volatilidad normal"
    BASELINE_LEN = 90 # 90 velas de 4h = 360h = 15 días
    
    # --------------------------------------------------------------------------
    # 2. CÁLCULO DE VOLATILIDAD ROBUSTA (Baseline)
    # --------------------------------------------------------------------------
    # ATR de 15 días para tener un umbral estable
    atr_long = ta.atr(df_macro['high'], df_macro['low'], df_macro['close'], length=BASELINE_LEN)
    
    # Umbral Dinámico:
    # Si en 5 días el precio se mueve más de 8 veces el ATR promedio de una vela de 4h, es significativo.
    # (El multiplicador 8 se ajusta para capturar tendencias claras de 5 días)
    dynamic_threshold = (atr_long / df_macro['close']) * 8 
    
    # --------------------------------------------------------------------------
    # 3. MIRAR AL FUTURO (Solo para Entrenamiento)
    # --------------------------------------------------------------------------
    # A. Retorno Neto Futuro
    future_close = df_macro['close'].shift(-LOOK_AHEAD)
    future_return = (future_close - df_macro['close']) / df_macro['close']
    
    # B. Volatilidad Futura (Rango High-Low relativo)
    # Calculamos el High máximo y Low mínimo de los próximos 5 días
    future_high = df_macro['high'].rolling(LOOK_AHEAD).max().shift(-LOOK_AHEAD)
    future_low = df_macro['low'].rolling(LOOK_AHEAD).min().shift(-LOOK_AHEAD)
    future_range_pct = (future_high - future_low) / df_macro['close']
    
    # C. INDICADOR EXTRA: Calidad de la Tendencia (Slope)
    # ta.slope calcula la pendiente de los últimos N periodos.
    # Queremos la pendiente de los PRÓXIMOS N periodos.
    # Por eso calculamos el slope y lo shifteamos hacia atrás.
    future_slope = ta.slope(df_macro['close'], length=LOOK_AHEAD).shift(-LOOK_AHEAD)
    
    # Normalizamos la pendiente para que sea comparable entre activos de distinto precio
    norm_slope = future_slope / df_macro['close']

    # --------------------------------------------------------------------------
    # 4. CLASIFICACIÓN
    # --------------------------------------------------------------------------
    
    # CONDICIÓN DE CAOS (Label 3)
    # El precio se movió muchísimo (Rango futuro > 3 veces lo normal), 
    # pero el retorno neto fue pequeño (fue y volvió -> Bart Simpson :) ).
    is_chaos = (future_range_pct > dynamic_threshold * 3.0) & (future_return.abs() < dynamic_threshold)
    
    # CONDICIÓN DE TENDENCIA (Label 1 y 2)
    # Regla reforzada: 
    # 1. El retorno supera el umbral dinámico (Fuerza)
    # 2. Y la pendiente confirma la dirección (Consistencia)
    
    # Umbral de pendiente mínima (para evitar subidas "lentas" que suelen revertir)
    slope_threshold = dynamic_threshold / LOOK_AHEAD 
    
    is_bull = (future_return > dynamic_threshold) & (norm_slope > slope_threshold * 0.5)
    is_bear = (future_return < -dynamic_threshold) & (norm_slope < -slope_threshold * 0.5)
    
    # ASIGNACIÓN DE PRIORIDADES
    conditions = [
        is_chaos, # Prioridad 1: Seguridad ante todo
        is_bull,  # Prioridad 2
        is_bear   # Prioridad 3
    ]
    choices = [3, 1, 2]
    
    # Default = 0 (Rango / Calma)
    df_macro['TARGET_CONTEXTO'] = np.select(conditions, choices, default=0)
    
    # --------------------------------------------------------------------------
    # 5. UNIFICAR A 5 MINUTOS (Broadcast)
    # --------------------------------------------------------------------------
    # Limpiar bordes vacíos del final (donde no hay futuro)
    df_macro.dropna(subset=['TARGET_CONTEXTO'], inplace=True)
    
    # Expandir la etiqueta de 4H a todas las velas de 5m correspondientes
    df_labels = df_macro['TARGET_CONTEXTO'].reindex(df.index, method='ffill')
    
    # Asignar al DataFrame original
    df['TARGET_CONTEXTO'] = df_labels
    
    return df

def etiquetar_regimen_tactico(df):
    """
    Etiquetado TÁCTICO (Corto Plazo).
    Timeframe: 1H
    Horizonte: 12 Horas (Optimizado)
    """
    # 1. RESAMPLE A 1H
    df_tactico = df.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    # ----------------------------------------------------------------------
    # 2. CONFIGURACIÓN OPTIMIZADA
    # ----------------------------------------------------------------------
    LOOK_AHEAD = 12  # <--- Predecimos solo medio día (12 velas)
    BASELINE_LEN = 168 # 1 Semana de referencia
    
    # 3. CÁLCULO DE VOLATILIDAD
    atr_long = ta.atr(df_tactico['high'], df_tactico['low'], df_tactico['close'], length=BASELINE_LEN)
    
    # Ajustamos el umbral dinámico.
    # Como el tiempo es la mitad (12 vs 24), el precio se mueve menos.
    # Bajamos el multiplicador de 4.0 a 3.0 para que sea sensible a movimientos de 12h.
    dynamic_threshold = (atr_long / df_tactico['close']) * 3.0 
    
    # 4. MIRAR AL FUTURO (12H)
    future_close = df_tactico['close'].shift(-LOOK_AHEAD)
    future_return = (future_close - df_tactico['close']) / df_tactico['close']
    
    future_high = df_tactico['high'].rolling(LOOK_AHEAD).max().shift(-LOOK_AHEAD)
    future_low = df_tactico['low'].rolling(LOOK_AHEAD).min().shift(-LOOK_AHEAD)
    future_range_pct = (future_high - future_low) / df_tactico['close']
    
    future_slope = ta.slope(df_tactico['close'], length=LOOK_AHEAD).shift(-LOOK_AHEAD)
    norm_slope = future_slope / df_tactico['close']

    # 5. CLASIFICACIÓN
    # CAOS: Rango muy grande pero retorno pobre
    is_chaos = (future_range_pct > dynamic_threshold * 2.0) & (future_return.abs() < dynamic_threshold)
    
    slope_threshold = dynamic_threshold / LOOK_AHEAD 
    
    is_bull = (future_return > dynamic_threshold) & (norm_slope > slope_threshold * 0.3)
    is_bear = (future_return < -dynamic_threshold) & (norm_slope < -slope_threshold * 0.3)
    
    conditions = [is_chaos, is_bull, is_bear]
    choices = [3, 1, 2] # 0 = Rango
    
    df_tactico['TARGET_TACTICO'] = np.select(conditions, choices, default=0)
    
    # 6. Broadcast a 5m
    df_tactico.dropna(subset=['TARGET_TACTICO'], inplace=True)
    df_labels = df_tactico['TARGET_TACTICO'].reindex(df.index, method='ffill')
    df['TARGET_TACTICO'] = df_labels
    
    return df