import numpy as np
import tensorflow as tf
import os
import glob
import sys
from sklearn.metrics import confusion_matrix

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_NPZ = os.path.join(BASE_DIR, 'DATOS_PARA_ENTRENAR_NPZ')
DIR_MODELS = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')

LABELS = ['RANGO', 'BULL', 'BEAR', 'CAOS']

# Umbrales actuales (para evaluar si filtran bien)
UMBRAL_MACRO = 0.60
UMBRALES_TACTICOS = {0: 0.34, 1: 0.30, 2: 0.32, 3: 0.38}

def evaluar_cerebro(nombre_humano, model_name, data_name, umbrales_dict=None, umbral_fijo=None):
    print(f"\n{'='*80}")
    print(f"🧠 EVALUANDO: {nombre_humano}")
    print(f"{'='*80}")
    
    path_m = os.path.join(DIR_MODELS, model_name)
    path_d = os.path.join(DIR_NPZ, data_name)
    
    if not os.path.exists(path_m): print("Falta modelo."); return

    # Cargar
    model = tf.keras.models.load_model(path_m)
    try:
        data = np.load(path_d)
        X = data['X']
        y_true = data['y']
    except:
        data = np.load(path_d, mmap_mode='r')
        X = data['X']
        y_true = data['y']
        
    # Validación (Último 20% - Futuro desconocido para la IA)
    split = int(len(y_true) * 0.8)
    X_val = X[split:]
    y_val = y_true[split:]
    
    print(f"   Muestras de prueba: {len(y_val)}")
    
    # Predicción
    probs = model.predict(X_val, batch_size=4096, verbose=0)
    pred_clase = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)
    
    # --- ANÁLISIS POR CLASE ---
    print(f"\n   📊 PRECISIÓN REAL (Usando tus umbrales configurados):")
    print(f"   {'CLASE':<10} | {'PREDICCIONES':<12} | {'ACIERTOS ✅':<12} | {'ERROR FATAL 💀':<15} | {'DESPERDICIO 💤'}")
    print("-" * 80)
    
    for i, label in enumerate(LABELS):
        # Definir el umbral a usar
        th = umbrales_dict[i] if umbrales_dict else umbral_fijo
        
        # Filtro de predicciones para esta clase con el umbral
        mask = (pred_clase == i) & (pred_conf >= th)
        total_pred = np.sum(mask)
        
        if total_pred == 0:
            print(f"   {label:<10} | 0            | -            | -               | -")
            continue
            
        # Analisis de la realidad de esas predicciones
        realidad = y_val[mask]
        
        aciertos = np.sum(realidad == i)
        pct_acierto = (aciertos / total_pred) * 100
        
        # Error Fatal: Dijo Bull y era Bear (o viceversa)
        # Rango y Caos no tienen "opuesto" tan claro, pero Bull(1) vs Bear(2) sí.
        fatal = 0
        if i == 1: fatal = np.sum(realidad == 2) # Dijo Bull, fue Bear
        elif i == 2: fatal = np.sum(realidad == 1) # Dijo Bear, fue Bull
        
        pct_fatal = (fatal / total_pred) * 100
        
        # Desperdicio: Dijo Tendencia y fue Rango (0)
        desperdicio = 0
        if i in [1, 2]: desperdicio = np.sum(realidad == 0)
        pct_desp = (desperdicio / total_pred) * 100
        
        
        str_acierto = f"{pct_acierto:.1f}%"
        str_fatal = f"{pct_fatal:.1f}%" if i in [1,2] else "-"
        str_desp = f"{pct_desp:.1f}%" if i in [1,2] else "-"
        
        calidad = ""
        if pct_acierto > 40: calidad = "✅ ÚTIL"
        elif pct_acierto > 25: calidad = "⚠️ AZAR"
        else: calidad = "❌ DAÑINO"
        
        if pct_fatal > 20: calidad = "💀 PELIGROSO"

        print(f"   {label:<10} | {total_pred:<12} | {str_acierto:<12} | {str_fatal:<15} | {str_desp}  {calidad}")

    print("\n   💡 GLOSARIO:")
    print("   - ACIERTOS: La IA tuvo razón. (El filtro ayudó).")
    print("   - ERROR FATAL: La IA dijo 'Subir' y el mercado se desplomó (o viceversa).")
    print("   - DESPERDICIO: La IA dijo 'Tendencia' y el mercado hizo Rango (Bloqueó por gusto).")

if __name__ == "__main__":
    # 1. MACRO (14 DÍAS)
    evaluar_cerebro("JEFE MACRO (14 Días)", 'model_IA_CONTEXTO.keras', 'dataset_IA_CONTEXTO.npz', umbral_fijo=UMBRAL_MACRO)
    
    # 2. TÁCTICO (1 HORA)
    evaluar_cerebro("DIRECTOR TÉCNICO (1 Hora)", 'model_IA_TACTICO_1H.keras', 'dataset_IA_TACTICO_1H.npz', umbrales_dict=UMBRALES_TACTICOS)