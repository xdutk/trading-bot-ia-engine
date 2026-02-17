import joblib
import numpy as np
import pandas as pd
import ccxt
import os

# --- RUTAS EXACTAS ---
# Este script va en la carpeta raíz del proyecto
PATH_MODEL = "MODELOS_ENTRENADOS/model_hmm_unsupervised.pkl"
PATH_SCALER = "DATOS_PARA_ENTRENAR_NPZ/scaler_hmm.pkl"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"

def check_hmm():
    print(f"🔬 INSPECCIONANDO CEREBRO HMM...")
    print(f"   📂 Modelo: {PATH_MODEL}")
    print(f"   ⚖️ Scaler: {PATH_SCALER}")

    # 1. Cargar Modelo y Scaler
    if not os.path.exists(PATH_MODEL) or not os.path.exists(PATH_SCALER):
        print("❌ ERROR CRÍTICO: No encuentro los archivos en las rutas especificadas.")
        print("   Verifica que estás corriendo el script desde la raíz del proyecto.")
        return

    try:
        model = joblib.load(PATH_MODEL)
        scaler = joblib.load(PATH_SCALER)
        print("✅ Archivos cargados correctamente.\n")
    except Exception as e:
        print(f"❌ Error cargando pkl: {e}")
        return

    # 2. Decodificar los Clusters (Medias del Modelo)
    # NOTA: Las medias del modelo están en ESCALA TRANSFORMADA. 
    
    print("📊 INTERPRETACIÓN DE CLUSTERS (Valores internos del modelo):")
    print(f"{'Cluster':<8} | {'Retorno (Tendencia)':<20} | {'Rango (Volatilidad)':<20}")
    print("-" * 60)
    
    means = model.means_
    for i in range(model.n_components):
        # Valor 0 = Retorno, Valor 1 = Rango
        tend = means[i][0]
        vol = means[i][1]
        
        # Interpretación relativa simplificada
        # generalmente:
        # Tendencia alta (+) = Bull, Baja (-) = Bear
        # Volatilidad alta = Caos/Trend, Baja = Rango
        
        print(f"C{i:<7} | {tend:>8.4f}             | {vol:>8.4f}")
    
    print("-" * 60)

    # 3. Obtener Datos del Mercado REAL AHORA
    print(f"\n🌍 MERCADO ACTUAL ({SYMBOL} {TIMEFRAME}):")
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=15)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        
        # Calculamos features
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['low']
        df.dropna(inplace=True)
        
        # Tomamos el último dato
        last_features = df[['log_ret', 'range']].iloc[-1].values.reshape(1, -1)
        
        # --- ESCALAR EL DATO ---

        last_features_scaled = scaler.transform(last_features)
        
        # Predicción
        current_state = model.predict(last_features_scaled)[0]
        
        # Probabilidades (Qué tan seguro está)
        probs = model.predict_proba(last_features_scaled)[0]
        prob_str = " | ".join([f"C{i}: {p*100:.1f}%" for i, p in enumerate(probs)])

        print(f"   Dato Crudo  -> Ret: {last_features[0][0]*100:.2f}% | Vol: {last_features[0][1]*100:.2f}%")
        print(f"   Dato Scaled -> Ret: {last_features_scaled[0][0]:.4f}  | Vol: {last_features_scaled[0][1]:.4f}")
        print(f"\n🤖 DIAGNÓSTICO FINAL:")
        print(f"   Estado Actual: Cluster {current_state}")
        print(f"   Confianza:     {prob_str}")
        
        if probs[current_state] > 0.999 and abs(last_features[0][0]) > 0.01:
            print("\n⚠️ ALERTA: Si la confianza es 100% clavada y el mercado se mueve mucho,")
            print("            podría haber un error de overfitting o escalado.")
        else:
            print("\n✅ El modelo parece estar respirando y calculando probabilidades.")

    except Exception as e:
        print(f"⚠️ Error procesando datos: {e}")

if __name__ == "__main__":
    check_hmm()