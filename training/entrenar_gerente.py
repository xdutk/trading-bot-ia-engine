import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
FILE_DATA = "DATASET_GERENTE_MASIVO_V3_CLEAN.csv"
MODEL_FILE = "MODELOS_ENTRENADOS/meta_model_manager_v2.pkl"
CONFIG_FILE = "MODELOS_ENTRENADOS/config_umbrales_pro.json" 

# "Zona de Francotirador" (Agresiva)
# WinRate alto para considerar que hay "mucha certeza"
TARGET_WINRATE_AGRESIVO = 0.60 

# Aquí definimos el "Filtro de Basura" (Defensivo)
# Queremos eliminar el fondo del barril, pero dejar pasar lo normal.
# "Eliminar todo lo que tenga probabilidad menor a X"
MIN_PROB_TOLERANCIA = 0.35 

# ==============================================================================
# 1. CARGA
# ==============================================================================
print(f"📂 Cargando datos limpios: {FILE_DATA} ...")
try:
    df = pd.read_csv(FILE_DATA)
except:
    print("❌ Error: No encuentro el CSV.")
    exit()

df = df.dropna()

# Encoding
df['strategy_id'] = df['strategy'].astype('category').cat.codes
df['side_id'] = df['side'].map({'BUY': 1, 'SELL': 0})

# Guardamos mapa
strat_categories = df['strategy'].astype('category').cat.categories
strat_map = dict(enumerate(strat_categories))

features = ['strategy_id', 'side_id', 'prob_ia', 
            'hmm_state', 'ctx_state', 'ctx_prob', 'tac_state', 'tac_prob', 
            'atr_pct', 'rsi', 'btc_trend', 'hour', 'day']

X = df[features]
y = df['target']

# ==============================================================================
# 2. ENTRENAMIENTO
# ==============================================================================
print(f"🧠 Entrenando Gerente V2 (Separando Longs/Shorts)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=600, learning_rate=0.015, max_depth=7, 
    subsample=0.75, colsample_bytree=0.75, eval_metric='logloss', use_label_encoder=False
)
model.fit(X_train, y_train)

# Predicciones
df_test = X_test.copy()
df_test['target_real'] = y_test
df_test['prob_gerente'] = model.predict_proba(X_test)[:, 1]
df_test['strategy_name'] = df_test['strategy_id'].map(strat_map)
df_test['side_name'] = df_test['side_id'].map({1: 'BUY', 0: 'SELL'})

# ==============================================================================
# 3. ANÁLISIS PROFUNDO
# ==============================================================================
print("\n" + "="*90)
print("🚦 REPORTE DE SEMÁFORO (VETO vs AGRESIVO)")
print("="*90)

config_final = {}

# Iteramos por Estrategia Y por Lado (Long/Short por separado)
for strat in df_test['strategy_name'].unique():
    config_final[strat] = {}
    
    for side in ['BUY', 'SELL']:
        subset = df_test[(df_test['strategy_name'] == strat) & (df_test['side_name'] == side)]
        
        if len(subset) < 20: 
            # Poca data, usamos valores por defecto seguros
            config_final[strat][side] = {"veto": 0.30, "agresivo": 0.70}
            continue

        wr_base = subset['target_real'].mean()
        
        print(f"\n🔹 {strat} {side} (Base WR: {wr_base:.1%})")
        
        # --- A. BUSCAR UMBRAL DE VETO (ROJO) ---
        # El veto debe ser bajo. Solo queremos bloquear si la IA dice "Esto es horrible" (ej: < 0.30)
        # Probamos cortes bajos. Si bloqueamos, ¿mejora el WR del resto?
        
        umbral_veto_elegido = 0.30 # Valor base conservador
        
        # Escaneamos la zona baja
        for cut in [0.20, 0.25, 0.30, 0.35, 0.40]:
            # Lo que sobrevive al veto
            survivors = subset[subset['prob_gerente'] >= cut]
            if len(survivors) == 0: continue
            
            wr_survivor = survivors['target_real'].mean()
            pct_bloqueado = 1 - (len(survivors) / len(subset))
            
            # Solo aceptamos el veto si bloquea menos del 50% de los trades
            # Y mejora el WR aunque sea un poco
            if pct_bloqueado < 0.50 and wr_survivor > wr_base:
                umbral_veto_elegido = cut
        
        print(f"   ⛔ VETO (< {umbral_veto_elegido:.2f}): Bloquea lo peor. El resto tiene WR mejorado.")

        # --- B. BUSCAR UMBRAL AGRESIVO ---
        # Aquí buscamos "Certeza". WinRate > 60%
        umbral_agg_elegido = 0.99 # Imposible por defecto
        
        for cut in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            snipers = subset[subset['prob_gerente'] >= cut]
            if len(snipers) < 5: continue # Muy pocos trades
            
            wr_sniper = snipers['target_real'].mean()
            
            if wr_sniper >= TARGET_WINRATE_AGRESIVO:
                umbral_agg_elegido = cut
                print(f"   🟢 AGRESIVO (> {cut:.2f}): Detectados {len(snipers)} trades con WR {wr_sniper:.1%}")
                break # Nos quedamos con el primero que cumpla (más volumen)
        
        if umbral_agg_elegido == 0.99:
            print("   ⚪ Sin zona agresiva clara (Mantenemos normal).")

        # Guardamos en la config
        config_final[strat][side] = {
            "veto": float(umbral_veto_elegido),
            "agresivo": float(umbral_agg_elegido)
        }

# ==============================================================================
# 4. GUARDADO
# ==============================================================================
joblib.dump(model, MODEL_FILE)
joblib.dump(strat_map, "MODELOS_ENTRENADOS/meta_model_strat_map.pkl")

with open(CONFIG_FILE, 'w') as f:
    json.dump(config_final, f, indent=4)

print("\n" + "="*90)
print(f"💾 CONFIGURACIÓN PRO GUARDADA: {CONFIG_FILE}")
print("El sistema ahora distingue entre Longs y Shorts y tiene 3 zonas de actuación.")