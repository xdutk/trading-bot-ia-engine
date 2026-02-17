import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import os
import glob

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
DIR_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATOS_PARA_ENTRENAR_NPZ')
DIR_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELOS_ENTRENADOS')

os.makedirs(DIR_MODELS, exist_ok=True)

# Parámetros
BATCH_SIZE = 2048 # Aumentado para velocidad (se suele usar millones de datos)
EPOCHS = 100

# ==============================================================================
# ARQUITECTURAS NEURONALES (DINÁMICAS)
# ==============================================================================

def construir_modelo_estrategia(shape_micro, shape_macro):
    """
    Modelo Híbrido: Micro (5m) + Macro (Diario).
    Para: FRPV, RANGO, BREAKOUT, TREND.
    """
    # Rama Micro (LSTM Rápida)
    in_mic = Input(shape=shape_micro, name='in_micro')
    x1 = LSTM(128, return_sequences=False)(in_mic)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    # Rama Macro (LSTM Lenta)
    in_mac = Input(shape=shape_macro, name='in_macro')
    x2 = LSTM(64, return_sequences=False)(in_mac)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Fusión
    combined = Concatenate()([x1, x2])
    
    # Cerebro
    z = Dense(64)(combined)
    z = LeakyReLU(negative_slope=0.1)(z)
    z = Dropout(0.2)(z)
    z = Dense(32)(z)
    z = LeakyReLU(negative_slope=0.1)(z)
    
    # Salida Binaria (Probabilidad de Ganar)
    out = Dense(1, activation='sigmoid', name='output')(z)
    
    model = Model(inputs=[in_mic, in_mac], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(name='precision')])
    
    return model, 'val_precision' # Precisión

def construir_modelo_contexto(shape_macro, n_classes):
    """
    Modelo Solo Macro.
    Para: IA_CONTEXTO (Jefe Macro).
    """
    in_mac = Input(shape=shape_macro, name='in_macro')
    
    # Red profunda para entender patrones complejos de largo plazo
    x = LSTM(128, return_sequences=True)(in_mac) 
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    z = Dense(64, activation='relu')(x)
    z = Dropout(0.2)(z)
    
    # Salida Multiclase
    out = Dense(n_classes, activation='softmax', name='output')(z)
    
    model = Model(inputs=in_mac, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model, 'val_accuracy'

def construir_modelo_tactico(shape_tactico, n_classes):
    """
    Modelo Solo Táctico (1H).
    Para: IA_TACTICO.
    """
    # LSTM Optimizada para secuencias medias (48 pasos)
    model = Sequential([
        Input(shape=shape_tactico),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, 'val_accuracy'

# ==============================================================================
# BUCLE DE ENTRENAMIENTO
# ==============================================================================
def entrenar_todos():
    print("🧠 INICIANDO FÁBRICA DE MODELOS (TRAIN ENSEMBLE)...")
    archivos = glob.glob(os.path.join(DIR_NPZ, "*.npz"))
    
    if not archivos: print(f"❌ No hay .npz en {DIR_NPZ}"); return

    for f in archivos:
        try:
            nombre_base = os.path.basename(f).replace('dataset_', '').replace('.npz', '')
            path_modelo = os.path.join(DIR_MODELS, f"model_{nombre_base}.keras")
            
            print(f"\n" + "="*60)
            print(f"🏗️  Entrenando Agente: {nombre_base}")
            print("="*60)
            
            # Cargar Datos
            try:
                data = np.load(f)
            except:
                data = np.load(f, mmap_mode='r') # Modo disco si explota la RAM

            # ------------------------------------------------------------------
            # DETECCIÓN DE TIPO DE MODELO
            # ------------------------------------------------------------------
            model = None
            monitor = ''
            X_train, X_val, y_train, y_val = [], [], [], []
            class_weights_dict = None

            # CASO A: TÁCTICO (1H)
            if 'TACTICO' in nombre_base:
                print("   Tipo: TÁCTICO (1H)")
                X = data['X']
                y = data['y']
                
                # Split
                split = int(len(y) * 0.8)
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                
                # Pesos (Importante para Táctico)
                pesos = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weights_dict = dict(enumerate(pesos))
                
                model, monitor = construir_modelo_tactico(X_train.shape[1:], len(np.unique(y)))
                
                inputs_train = X_train
                inputs_val = X_val

            # CASO B: CONTEXTO MACRO (14D)
            elif 'CONTEXTO' in nombre_base:
                print("   Tipo: CONTEXTO MACRO")
                X = data['X']
                y = data['y']
                
                split = int(len(y) * 0.8)
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                
                # Pesos
                pesos = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weights_dict = dict(enumerate(pesos))
                
                model, monitor = construir_modelo_contexto(X_train.shape[1:], len(np.unique(y)))
                
                inputs_train = X_train
                inputs_val = X_val

            # CASO C: ESTRATEGIAS
            else:
                print("   Tipo: ESTRATEGIA (Micro+Macro)")
                X_mic = data['X_micro']
                X_mac = data['X_macro']
                y = data['y']
                
                split = int(len(y) * 0.8)
                X_mic_train, X_mic_val = X_mic[:split], X_mic[split:]
                X_mac_train, X_mac_val = X_mac[:split], X_mac[split:]
                y_train, y_val = y[:split], y[split:]
                
                # Pesos (Binario: 0 vs 1)
                # !!!: Si ganamos poco (WR bajo), el peso del 1 será alto. Eso ayuda a encontrar oportunidades.
                pesos = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weights_dict = dict(enumerate(pesos))
                
                model, monitor = construir_modelo_estrategia(X_mic_train.shape[1:], X_mac_train.shape[1:])
                
                inputs_train = [X_mic_train, X_mac_train]
                inputs_val = [X_mic_val, X_mac_val]

            print(f"   Muestras Train: {len(y_train)} | Val: {len(y_val)}")
            print(f"   Pesos: {class_weights_dict}")

            # ------------------------------------------------------------------
            # ENTRENAMIENTO
            # ------------------------------------------------------------------
            callbacks = [
                ModelCheckpoint(path_modelo, save_best_only=True, monitor=monitor, mode='max', verbose=1),
                EarlyStopping(monitor=monitor, patience=10, restore_best_weights=True, mode='max'),
                ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=4, mode='max', min_lr=1e-6)
            ]
            
            model.fit(
                inputs_train, y_train,
                validation_data=(inputs_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                class_weight=class_weights_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            print(f"✅ Guardado: {path_modelo}")
            
        except Exception as e:
            print(f"❌ Error entrenando {os.path.basename(f)}: {e}")

    print("\n🏁 PROCESO FINALIZADO.")

if __name__ == "__main__":
    entrenar_todos()