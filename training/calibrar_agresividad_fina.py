import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
FILE_DATA = "DATASET_GERENTE_MASIVO_V3_CLEAN.csv"

# ==============================================================================
# 1. CARGA Y PREPARACIÓN
# ==============================================================================
print(f"📂 Cargando datos: {FILE_DATA} ...")
try:
    df = pd.read_csv(FILE_DATA)
except:
    print("❌ Error: No encuentro el CSV.")
    exit()

df = df.dropna()

# Encoding y Features
df['strategy_id'] = df['strategy'].astype('category').cat.codes
df['side_id'] = df['side'].map({'BUY': 1, 'SELL': 0})
strat_categories = df['strategy'].astype('category').cat.categories
strat_map = dict(enumerate(strat_categories))

features = ['strategy_id', 'side_id', 'prob_ia', 
            'hmm_state', 'ctx_state', 'ctx_prob', 'tac_state', 'tac_prob', 
            'atr_pct', 'rsi', 'btc_trend', 'hour', 'day']

X = df[features]
y = df['target']

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=6, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predicciones
df_test = X_test.copy()
df_test['target_real'] = y_test
df_test['prob_gerente'] = model.predict_proba(X_test)[:, 1]
df_test['strategy_name'] = df_test['strategy_id'].map(strat_map)
df_test['side_name'] = df_test['side_id'].map({1: 'BUY', 0: 'SELL'})

# ==============================================================================
# 2. ANÁLISIS DE SENSIBILIDAD DE AGRESIVIDAD (ZONA VERDE)
# ==============================================================================
print("\n" + "="*100)
print("🟢 ANÁLISIS DE AGRESIVIDAD: CALIBRANDO EL POTENCIAL DE GANANCIA")
print("="*100)
print("Leyenda:")
print(" - % Sobrevive: Porcentaje de trades que entran en la Zona Verde (> Umbral).")
print(" - WR Agresivo: WinRate de esos trades de alta certeza (¡QUEREMOS > 60%!).")
print("-" * 100)

for strat in df_test['strategy_name'].unique():
    for side in ['BUY', 'SELL']:
        subset = df_test[(df_test['strategy_name'] == strat) & (df_test['side_name'] == side)]
        if len(subset) < 20: continue
        
        print(f"\n🔹 ESTRATEGIA: {strat} {side} (Total Oportunidades: {len(subset)})")
        print(f"   {'Umbral':<8} | {'% Sobrevive':<15} | {'WR Agresivo':<15} | {'Trades Aprobados':<18}")
        print("   " + "-"*70)
        
        # Probamos umbrales de 0.50 (moneda al aire) hasta 0.90 (casi imposible)
        for corte in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            # SOBREVIVE: trades con prob_gerente >= corte
            sobrevivientes = subset[subset['prob_gerente'] >= corte]
            
            if len(sobrevivientes) == 0: continue
            
            pct_sobrevive = len(sobrevivientes) / len(df_test) # % del total del dataset
            wr_agresivo = sobrevivientes['target_real'].mean()
            
            # MARCADOR: Si el WR es superior al 60%, es una buena zona.
            marker = "✅" if wr_agresivo >= 0.60 else "⚠️" if wr_agresivo >= 0.55 else "❌"
            
            print(f"   > {corte:.2f}  | {pct_sobrevive:.3%}      | {wr_agresivo:.1%} {marker:<10} | {len(sobrevivientes)}")

print("\n" + "="*100)
print("💡 CÓMO ELEGIR EL PUNTO AGRESIVO (Trade-Off):")
print("1. Empieza en el Umbral más bajo que cumpla el 60% (✅).")
print("2. Si hay dos Umbrales que cumplen, elige el MÁS BAJO para capturar más Trades.")
print("   Ej: Si 0.60 da 70% WR (20 trades) y 0.70 da 75% WR (5 trades), elegimos 0.60.")
print("3. Si la estrategia NO tiene un punto ✅, se queda en el Umbral 0.99 (Inactivo).")