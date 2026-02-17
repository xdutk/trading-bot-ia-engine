import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import json
import pandas_ta as ta
from datetime import datetime, timedelta
import threading
import sys
import traceback
import warnings
import requests
import uuid
from dotenv import load_dotenv
import psutil
import shutil
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL
# ------------------------------------------------------------------------------
# warnings.filterwarnings("ignore") 
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Carga .env
load_dotenv()

# --- TELEGRAM ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN') 
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ ERROR CRÍTICO: Faltan credenciales de Telegram en el .env")
    sys.exit()

USAR_TELEGRAM = True

# ==============================================================================
# 2. ERRORES (TELEGRAM)
# ==============================================================================
def reportar_crash_telegram(exc_type, exc_value, exc_tb):
    """Captura crash y avisar a Telegram"""
    if issubclass(exc_type, KeyboardInterrupt): sys.__excepthook__(exc_type, exc_value, exc_tb); return
    error_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))[-4000:]
    msg_fatal = f"💀 <b>CRASH FATAL</b>\n<pre>{error_trace}</pre>"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg_fatal, "parse_mode": "HTML"}, timeout=5)
    except: pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = reportar_crash_telegram

# ==============================================================================
# 3. IMPORTACIÓN DE MÓDULOS
# ==============================================================================
try:
    from MODULOS import market_context
    from ESTRATEGIAS import frpv, mean_reversion, breakout, simple_trend
except ImportError as e:
    print(f"❌ Error importando: {e}"); sys.exit()

# Configuración General y Futuros
PAPER_TRADING = True # <--- FALSE PARA DINERO REAL
API_KEY = os.getenv('BINANCE_API_KEY')      
API_SECRET = os.getenv('BINANCE_SECRET_KEY')

# Validación de seguridad
if not API_KEY or not API_SECRET:
    print("❌ ERROR CRÍTICO: No se encontraron las API KEYS en el archivo .env")
    exit()

# VARIABLES DE CONTROL GLOBAL
bot_running = True   # False, el script se apaga.
bot_paused = False   # True, no abre nuevas operaciones (gestiona las viejas)
ultimo_update_id = 0 
SILENCIAR_AUTO_SYNC = False

COINS_TO_TRADE = [
    # --- (Estabilidad) ---
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
    
    # --- (Volumen) ---
    'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 
    'LINK/USDT', 'DOT/USDT', 'LTC/USDT',
    
    # --- VOLATILIDAD / TRENDING ---
    'SUI/USDT',   # Fuerte recientemente
    'APT/USDT',   
    'FET/USDT',   # Sector IA
    'RENDER/USDT',# Sector IA
    '1000PEPE/USDT',  # Meme 
    'WIF/USDT',   # Meme
    'NEAR/USDT',  # L1 sólida
    'INJ/USDT',   # DeFi
    'TIA/USDT',   # Modular blockchain
    'OP/USDT',    # L2
    'ZEC/USDT',
    'DASH/USDT'    
]

TOP20_SYMBOLS = [
    # --- TOP 5 ---
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',

   
    # --- TOP 10-15 ---
    'DOGE/USDT',  
    'ADA/USDT',   # Cardano
    'TRX/USDT',   # Tron (Mucho volumen real)
    'AVAX/USDT',  # Avalanche
    'TON/USDT',   # Toncoin
    
    # --- INFRAESTRUCTURA Y DEFI (Sólidas) ---
    'DOT/USDT',   # Polkadot
    'LINK/USDT',  # Chainlink
    'BCH/USDT',   # Bitcoin Cash
    'LTC/USDT',   # Litecoin
    'UNI/USDT',   # Uniswap
    
    # --- L1s y L2s FUERTES ---
    'NEAR/USDT',  # Near Protocol
    'POL/USDT',   # Polygon
    'APT/USDT',   # Aptos
    'SUI/USDT',   # Sui
    'XLM/USDT'    # Stellar
]

gerente = None
strat_map_inv = {}
umbrales_pro = {}

TIMEFRAME = '5m'
TIMEFRAME_MACRO = '1d' 
DAYS_HISTORY = 300 

# PUNTOS 2, 9: Gestión de Capital y Cooldown
start_time = datetime.now()
CAPITAL_MAXIMO = 40.0
PORCENTAJE_BASE = 0.30 

# --- LÍMITES DINÁMICOS ---
MAX_DAILY_LOSS_PCT = 0.05   # Default: 5% del Capital Máximo
MAX_SPREAD_ALLOWED = 0.002  # Default: 0.2% de diferencia Bid/Ask

BENCHMARKS_ATR = {
    'FRPV':     1.0,
    'TREND':    1.0,
    'BREAKOUT': 0.4,
    'RANGO':    0.6,
    'DEFAULT':  3.0
}

# --- AJUSTES DEL SESGO ---
SESGO_WINDOW_TRADES = 15       # Cuántos trades miramos hacia atrás
SESGO_PERDON_MINUTOS = 240     # Minutos para levantar el "cooldown" (4 horas)

LEVERAGE_BASE = 3       
LEVERAGE_MAX = 24     
# --- ESCALERA DE APALANCAMIENTO ---
LEVERAGE_STEPS = [1, 3, 6, 12, 24] 

# COOLDOWNS (Punto 7)
MAX_FAILURES_STRAT = 3      # Si una estrategia falla 3 veces seguidas (globalmente)...
COOLDOWN_GLOBAL_CANDLES = 30 # ...se apaga para TODOS los activos por 30 velas (2.5h)
COOLDOWN_CANDLES = 25       # Cooldown individual

# NUEVOS LÍMITES (Punto 4)
MAX_TRADES_GLOBAL = 8       # Techo total de la cuenta
MAX_TRADES_STRAT = 3        # Máximo por tipo (ej: solo 3 TREND a la vez)
MAX_TRADES_SIDE = 5         # (Máximo 5 Longs o 5 Shorts)

# GESTIÓN DE SALIDA (Punto 8)
BE_TRIGGER_RATIO = 0.80     # Al 80% del recorrido al TP, SL se mueve al Break Even

# UMBRALES (Punto 5 y 6)
UMBRAL_CONTEXTO = 0.54      # Umbal de confianza mínima del Jefe
UMBRALES_TACTICOS = {0: 0.34, 1: 0.30, 2: 0.32, 3: 0.38}

# VIP SETTINGS (Punto 9 y 10)
VIP_WINDOW_CANDLES = 60     # Duración del "pase VIP" tras ganar
VIP_UMBRAL_DISCOUNT = 0.05  # Cuánto bajamos la exigencia (ej: 0.68 -> 0.63)

#  PUNTO 4: Reglas HMM (VIP)
# 'NOMBRE' -> Long y Short.
# 'NOMBRE_SHORT' -> Permite Short.
# 'NOMBRE_LONG' -> Permite Long.

REGLAS_HMM = {
    # CLUSTER 0: Bajista Fuerte
    0: ['RANGO_SHORT', 'BREAKOUT_SHORT', 'TREND_SHORT', 'FRPV_SHORT'], 

    # CLUSTER 1: Alcista Fuerte
    1: ['RANGO', 'BREAKOUT_LONG', 'TREND_LONG', 'FRPV_LONG'], 

    # CLUSTER 2: Rango Calmo
    2: ['RANGO', 'BREAKOUT_SHORT', 'FRPV'],

    # CLUSTER 3: Volatilidad
    3: [] 
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_MODELOS = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')
DIR_SCALERS = os.path.join(BASE_DIR, 'DATOS_PARA_ENTRENAR_NPZ')
LOG_FILE = os.path.join(BASE_DIR, 'live_trades_log.txt')
STATE_FILE = os.path.join(BASE_DIR, 'bot_state_v2.json') 

CONFIG_STRAT = {
    'FRPV':     {'tp': 14.0, 'sl': 2.0, 'buy': 'Real Price Buy', 'sell': 'Real Price Sell', 'umbral': 0.50},
    'RANGO':    {'tp': 3.5, 'sl': 1.5, 'buy': 'Real_Price_Rango_Buy', 'sell': 'Real_Price_Rango_Sell', 'umbral': 0.58},
    'BREAKOUT': {'tp': 4.0, 'sl': 1.0, 'buy': 'Real_Price_Breakout_Buy', 'sell': 'Real_Price_Breakout_Sell', 'umbral': 0.65},
    'TREND':    {'tp': 7.0, 'sl': 2.0, 'buy': 'Real_Price_Trend_Buy', 'sell': 'Real_Price_Trend_Sell', 'umbral': 0.63}
}

COLS_MICRO = ['f_5m_ret', 'f_5m_vol_z', 'f_dist_kama', 'f_dist_sma', 'f_dist_lrc', 'f_5m_overext']
COLS_MACRO = ['Retorno Diario Activo', 'Retorno Top20', 'Diferencia vs Mercado', 'Promedio Últimos N', 'Z-Score Sesgo Ponderado', 'Correlación', 'Z-Score Volumen', 'Vol_Signal', 'BTC_Trend_Score', 'BTC_Daily_Ret']

# PUNTO 7
bot_state = {"active_trades": {}, "strategy_state": {}} 

# --- THREAD LOCK ---
# Un hilo de escritura a la vez 
lock_estado = threading.Lock() 

# Para la memoria RAM
memory_lock = threading.Lock() 
# -------------------------------------------------

# ==============================================================================
# GESTIÓN DE ESTADO Y APALANCAMIENTO (PUNTO 2)
# ==============================================================================
def inicializar_stats_globales(strat_name):
    """Inicializa el contador global de fallos para una estrategia"""
    if "global_strat_perf" not in bot_state:
        bot_state["global_strat_perf"] = {}
    
    if strat_name not in bot_state["global_strat_perf"]:
        bot_state["global_strat_perf"][strat_name] = {
            "fails": 0,             # Pérdidas consecutivas globales
            "cooldown_until": None  # Si está castigada globalmente
        }

def inicializar_estado_estrategia(key):
    """Inicializa el estado individual por par/estrategia"""
    if key not in bot_state["strategy_state"]:
        bot_state["strategy_state"][key] = {
            "leverage": LEVERAGE_BASE, 
            "consecutive_losses": 0, 
            "recovery_wins": 0, 
            "cooldown_until": None,
            "status": "NORMAL",
            "vip_until": None
        }
    else:
        if "vip_until" not in bot_state["strategy_state"][key]:
            bot_state["strategy_state"][key]["vip_until"] = None

def guardar_estado():
    global bot_state
    temp_file = STATE_FILE + ".tmp"
    
    try:
        # Copia segura de la memoria
        with memory_lock:
            # Esto evita que active_trades cambie mientras se guarda
            data_to_save = json.loads(json.dumps(bot_state, default=str))

        # Escribir en disco
        with lock_estado:
            with open(temp_file, 'w') as f: 
                json.dump(data_to_save, f, indent=4)
            
            if os.path.exists(temp_file):
                os.replace(temp_file, STATE_FILE)
            
    except Exception as e:
        print(f"⚠️ Error guardando estado: {e}")

def cargar_estado():
    global bot_state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                content = f.read().strip()
                if not content: 
                    # Archivo existe pero está vacío
                    bot_state = {
                        "active_trades": {}, 
                        "strategy_state": {}, 
                        "global_strat_perf": {},
                        "closed_history": [],
                        "volatility_blocklist": {},
                        "ai_views": {} 
                    }
                else:
                    # Carga exitosa
                    bot_state = json.loads(content)
                    
                    # 1. Asegurar Global Perf
                    if "global_strat_perf" not in bot_state:
                        bot_state["global_strat_perf"] = {}
                    
                    # 2. Asegurar Historial Cerrado
                    if "closed_history" not in bot_state:
                        bot_state["closed_history"] = [] 

                    # 3. Asegurar Blocklist de Volatilidad
                    if "volatility_blocklist" not in bot_state:
                        bot_state["volatility_blocklist"] = {}
                        
                    # 4. Asegurar Vistas de IA
                    if "ai_views" not in bot_state:
                        bot_state["ai_views"] = {}
                    
                    print(f"🔄 Estado V3.0 recuperado: {len(bot_state['active_trades'])} trades activos.")
                    
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Archivo corrupto ({e}). Iniciando limpio.")
            # Error de lectura
            bot_state = {
                "active_trades": {}, 
                "strategy_state": {}, 
                "global_strat_perf": {},
                "closed_history": [],
                "volatility_blocklist": {},
                "ai_views": {}
            }
    else:
        # No existe el archivo
        bot_state = {
            "active_trades": {}, 
            "strategy_state": {}, 
            "global_strat_perf": {},
            "closed_history": [],
            "volatility_blocklist": {},
            "ai_views": {}
        }

def log_to_file(msg):
    try:
        with open(LOG_FILE, "a", encoding='utf-8') as f: f.write(f"{datetime.now()} - {msg}\n")
    except: pass

def log_trade_to_file(symbol, strategy, side, exit_reason, pnl_amount, pnl_pct):
    """Guarda operación con etiquetas en live_trades_log.txt"""
    filename = "live_trades_log.txt"
    timestamp = datetime.now().strftime('%d/%m %H:%M')
    
    # 1. Distinguir el MODO
    modo_txt = "[PAPER]" if PAPER_TRADING else "[REAL]"

    # 2. Etiquetas visuales
    if pnl_amount > 0:
        icon, tag = "✅", "WIN"
    elif pnl_amount < 0:
        if abs(pnl_pct) < 0.15: icon, tag = "🛡️", "BE-" # (fees)
        else: icon, tag = "❌", "LOSS"
    else:
        icon, tag = "😐", "NEUTRAL"

    # 3. Construir línea
    line = f"{timestamp} | {modo_txt} | {icon} {tag} ({exit_reason}) | {symbol} {side} | {strategy} | PnL: {pnl_pct:.2f}% (${pnl_amount:.2f})\n"
    
    try:
        with open(filename, "a", encoding="utf-8") as f: f.write(line)
    except: pass

def display_sym(s):
    """Helper visual para limpiar el :USDT"""
    try:
        return s.split(':')[0].replace('/USDT', '')
    except:
        return s

def obtener_historial_telegram(n=10):
    """Lee las últimas n líneas del log para Telegram."""
    filename = "live_trades_log.txt"
    if not os.path.exists(filename): return "📂 Sin historial aún."
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [l for l in f.readlines() if l.strip()]
        last = lines[-n:][::-1]
        if not last: return "📭 Historial vacío."
        return f"📜 <b>ÚLTIMAS {len(last)} OPERACIONES</b>\n\n" + "".join(last)
    except Exception as e: return f"⚠️ Error leyendo: {e}"

# ==============================================================================
# LÓGICA DE NEGOCIO
# ==============================================================================

def actualizar_gestion_capital(key, res, strategy_name=None):
    """
    Maneja Apalancamiento usando la escalera.
    Ignora MANUAL/LIMIT para no ensuciar stats.
    """
    
    # 1. Detectar estrategia
    if not strategy_name:
        try: strategy_name = key.split('_')[-1]
        except: strategy_name = "UNKNOWN"

    # 2. IGNORAR Manuales o Limits
    # Si es un trade manual o limit, que no afecte el apalancamiento del bot
    if "MANUAL" in strategy_name or "LIMIT" in strategy_name or "LIMIT_SNIPER" in strategy_name:
        return 

    inicializar_stats_globales(strategy_name)

    if res == 'NEUTRAL': 
        inicializar_estado_estrategia(key)
        st = bot_state["strategy_state"][key]
        cooldown_tactico = datetime.now() + timedelta(minutes=15)
        st["cooldown_until"] = str(cooldown_tactico)
        bot_state["strategy_state"][key] = st
        guardar_estado()
        print(f"❄️ COOLDOWN TÁCTICO: {key} pausada 15m por Break Even.")
        return
    
    st = bot_state["strategy_state"][key]
    lev_actual = st["leverage"]
    status = st.get("status", "NORMAL")
    
    try: idx_actual = LEVERAGE_STEPS.index(lev_actual)
    except ValueError: idx_actual = 1 
    
    if res == 'WIN':
        st["consecutive_losses"] = 0
        st["vip_until"] = str(datetime.now() + timedelta(minutes=5*VIP_WINDOW_CANDLES))
        bot_state["global_strat_perf"][strategy_name]["fails"] = 0

        if status == "PENALTY":
            st["recovery_wins"] += 1
            if st["recovery_wins"] >= 2:
                st["leverage"] = LEVERAGE_BASE 
                st["status"] = "NORMAL"
                st["recovery_wins"] = 0
        elif status == "RECOVERING":
            st["recovery_wins"] += 1
            if st["recovery_wins"] >= 1:
                st["status"] = "NORMAL"
                st["recovery_wins"] = 0
        elif status == "NORMAL":
            new_idx = min(idx_actual + 1, len(LEVERAGE_STEPS) - 1)
            st["leverage"] = LEVERAGE_STEPS[new_idx]

    else: # LOSS
        st["consecutive_losses"] += 1
        st["recovery_wins"] = 0
        st["cooldown_until"] = str(datetime.now() + timedelta(minutes=5*COOLDOWN_CANDLES))
        
        bot_state["global_strat_perf"][strategy_name]["fails"] += 1
        if bot_state["global_strat_perf"][strategy_name]["fails"] >= MAX_FAILURES_STRAT:
            tiempo_cool = str(datetime.now() + timedelta(minutes=5*COOLDOWN_GLOBAL_CANDLES))
            bot_state["global_strat_perf"][strategy_name]["cooldown_until"] = tiempo_cool
            print(f"🛑 GLOBAL COOL-DOWN: {strategy_name} pausada.")
        
        if st["consecutive_losses"] >= 2:
            st["leverage"] = 1 
            st["status"] = "PENALTY"
        else:
            new_idx = max(idx_actual - 1, 0)
            st["leverage"] = LEVERAGE_STEPS[new_idx]
            st["status"] = "RECOVERING"

    bot_state["strategy_state"][key] = st
    guardar_estado()
    
def check_cooldown(key):
    inicializar_estado_estrategia(key); until = bot_state["strategy_state"][key]["cooldown_until"]
    return True if until and datetime.now() < datetime.fromisoformat(until) else False

def inicializar_exchange():
    # Punto 1: Futuros
    return ccxt.binance({
        'apiKey': API_KEY, 
        'secret': API_SECRET, 
        'enableRateLimit': True, 
        'options': {
            'defaultType': 'future', 
            'adjustForTimeDifference': True,
            'warnOnFetchOpenOrdersWithoutSymbol': False 
        }
    })

def enviar_telegram(mensaje):
    """Envía notificaciones a Telegram sin bloquear el bot"""
    if not USAR_TELEGRAM: return
    
    def _send():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "HTML"}
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"⚠️ Error Telegram: {e}")

    # En un hilo separado
    threading.Thread(target=_send).start()

def escuchar_telegram(exchange):
    """Escucha comandos desde Telegram"""

    # --- GLOBALES ---

    global bot_running, bot_paused, ultimo_update_id, CAPITAL_MAXIMO, PAPER_TRADING, PORCENTAJE_BASE
    global gerente, umbrales_pro, strat_map_inv, bot_state, BENCHMARKS_ATR, STATE_FILE
    global MAX_DAILY_LOSS_PCT, MAX_SPREAD_ALLOWED 
    global SILENCIAR_AUTO_SYNC
    
    if not USAR_TELEGRAM: return
    
    print("👂 Escuchando comandos de Telegram...")
    url_base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    
    startup_time = int(time.time()) 
    
    while bot_running:
        try:
            params = {"offset": ultimo_update_id + 1, "timeout": 10}
            response = requests.get(url_base, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    for update in data["result"]:
                        ultimo_update_id = update["update_id"]
                        
                        if "message" in update and "text" in update["message"]:
                            
                            msg_date = update["message"].get("date", 0)
                            if msg_date < startup_time: continue 

                            chat_id = str(update["message"]["chat"]["id"])
                            texto = update["message"]["text"].lower().strip()
                            
                            if chat_id != TELEGRAM_CHAT_ID:
                                print(f"⚠️ Intento de comando no autorizado: {chat_id}")
                                continue

                            print(f"📩 Comando Telegram recibido: {texto}")

                            if texto == '/status':
                                try:
                                    # 1. Cálculo de Uptime
                                    ahora = datetime.now()
                                    tiempo_run = ahora - start_time
                                    dias = tiempo_run.days
                                    horas, resto = divmod(tiempo_run.seconds, 3600)
                                    minutos, segundos = divmod(resto, 60)
                                    str_uptime = f"{dias}d {horas}h {minutos}m"

                                    # 2. Recursos (RAM/CPU)
                                    try:
                                        process = psutil.Process(os.getpid())
                                        mem_mb = process.memory_info().rss / 1024 / 1024
                                        cpu_uso = psutil.cpu_percent()
                                        info_recursos = f"💾 RAM: {mem_mb:.1f} MB | ⚡ CPU: {cpu_uso}%"
                                    except:
                                        info_recursos = "💾 RAM: (No disponible)"

                                    # 3. Estados Variables
                                    estado_paper = "PAPER 📄" if PAPER_TRADING else "REAL 💵"
                                    estado_vital = "DORMIDO 💤" if bot_paused else "ACTIVO ⚡"
                                    estado_sync = "SILENCIO 🔕" if SILENCIAR_AUTO_SYNC else "ACTIVO 🔔"
                                    
                                    # 4. Contexto de Mercado
                                    cluster_id = bot_state.get("market_cluster", "?")
                                    txt_cluster = f"C{cluster_id}"
                                    
                                    # 5. Datos Financieros
                                    num_trades = len(bot_state["active_trades"])
                                    base_usd = CAPITAL_MAXIMO * PORCENTAJE_BASE

                                    msg_status = (
                                        f"🤖 <b>ESTADO DEL SISTEMA</b>\n"
                                        f"━━━━━━━━━━━━━━━━\n"
                                        f"🚦 Estado: <b>{estado_vital}</b>\n"
                                        f"⏱️ Uptime: {str_uptime}\n"
                                        f"⚙️ Modo: <b>{estado_paper}</b>\n"
                                        f"📡 Auto-Sync: {estado_sync}\n"
                                        f"🌎 Contexto: <b>{txt_cluster}</b>\n"
                                        f"━━━━━━━━━━━━━━━━\n"
                                        f"📊 Trades Abiertos: {num_trades}\n"
                                        f"💰 Cap Max: ${CAPITAL_MAXIMO:.2f}\n"
                                        f"🌱 Cap Base: ${base_usd:.2f} ({int(PORCENTAJE_BASE*100)}%)\n" 
                                        f"━━━━━━━━━━━━━━━━\n"
                                        f"{info_recursos}"
                                    )
                                    enviar_telegram(msg_status)
                                
                                except Exception as e:
                                    print(f"❌ Error al generar Status: {e}")
                                    enviar_telegram(f"⚠️ Error interno en /status: {e}")

                            # --- DESCARGAR JSON ---
                            elif texto in ['/json', '/debug', 'debug']:
                                try:
                                    if os.path.exists(STATE_FILE):
                                        enviar_telegram("📂 <b>Enviando cerebro...</b>")
                                        # API de Telegram para mandar archivos
                                        url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
                                        with open(STATE_FILE, 'rb') as f:
                                            requests.post(url_doc, data={"chat_id": TELEGRAM_CHAT_ID}, files={"document": f})
                                    else:
                                        enviar_telegram("⚠️ El archivo .json aún no se ha creado (espera al primer trade).")
                                except Exception as e:
                                    enviar_telegram(f"❌ Error enviando JSON: {e}")

                            # --- SYNC (Auditoría Total) ---
                            elif texto in ['/sync', 'sincronizar', 'clean', 'audit']:
                                enviar_telegram("🕵️ <b>AUDITORÍA PROFUNDA INICIADA...</b>\nRevisando fantasmas y posiciones desnudas...")
                                try:
                                    # Texto completo con el reporte
                                    reporte_final = sincronizar_cartera_real(exchange)
                                    enviar_telegram(reporte_final)
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Sync: {e}")
                            
                            elif texto == '/config':
                                base_usd = CAPITAL_MAXIMO * PORCENTAJE_BASE
                                msg = (
                                    f"⚙️ <b>CONFIGURACIÓN ACTUAL</b>\n\n"
                                    f"💰 Cap Max: <b>${CAPITAL_MAXIMO:.2f}</b>\n"
                                    f"🌱 Base: <b>{int(PORCENTAJE_BASE*100)}%</b> (${base_usd:.2f})\n\n"
                                    f"📏 <b>Techos (Benchmarks ATR):</b>\n"
                                )
                                for s, v in BENCHMARKS_ATR.items():
                                    msg += f" • {s}: {v}\n"
                                
                                enviar_telegram(msg)

                            elif texto.startswith('/setbase'):
                                try:
                                    
                                    nuevo_pct = float(texto.split()[1])
                                    if 0.01 <= nuevo_pct <= 1.0:
                                        PORCENTAJE_BASE = nuevo_pct
                                        base_usd = CAPITAL_MAXIMO * PORCENTAJE_BASE
                                        enviar_telegram(f"✅ <b>BASE ACTUALIZADA</b>\nNuevo %: {int(PORCENTAJE_BASE*100)}%\nEn USD: ${base_usd:.2f}")
                                    else:
                                        enviar_telegram("⚠️ Error. Usa un valor entre 0.01 y 1.0")
                                except:
                                    enviar_telegram("⚠️ Error. Ejemplo: /setbase 0.15")

                            elif texto.startswith('/setbench'):
                                try:
                                    
                                    parts = texto.split()
                                    strat = parts[1].upper()
                                    val = float(parts[2])
                                    
                                    if strat in BENCHMARKS_ATR:
                                        BENCHMARKS_ATR[strat] = val
                                        enviar_telegram(f"✅ <b>BENCHMARK ACTUALIZADO</b>\n{strat} ahora busca {val} ATRs de ganancia.")
                                    else:
                                        enviar_telegram(f"⚠️ Estrategia no encontrada. Disponibles: {list(BENCHMARKS_ATR.keys())}")
                                except:
                                    enviar_telegram("⚠️ Error. Ejemplo: /setbench TREND 1.5")
                            
                            # --- RESETEAR SESGO/HISTORIAL ---
                            elif texto in ['/resetsesgo', '/resetbias', 'resetbias']:
                                try:
                                    
                                    bot_state["closed_history"] = [] 
                                    guardar_estado()
                                    
                                    msg = (
                                        "⚖️ <b>SESGO REINICIADO</b>\n"
                                        "El historial de aprendizaje ha sido borrado.\n"
                                        "El bot operará con umbrales NEUTROS hasta tener nueva data."
                                    )
                                    enviar_telegram(msg)
                                    print(f"\n⚖️ Historial borrado por comando Telegram. Sesgo Neutro.\n")
                                except Exception as e:
                                    enviar_telegram(f"❌ Error al resetear sesgo: {e}")

                            # --- PNL (SOLO DINERO REAL) ---
                            elif texto in ['/pnl', 'ganancias', 'profit', 'resultados']:
                                try:
                                    historial = bot_state.get("closed_history", [])
                                    if not historial:
                                        enviar_telegram("📭 <b>Sin historial.</b>")
                                        continue

                                    total_usd = 0.0
                                    diario_usd = 0.0
                                    wins = 0
                                    real_trades_count = 0 # Contador solo de reales
                                    today_str = datetime.now().strftime('%Y-%m-%d')

                                    for t in historial:
                                        # FILTRO
                                        # (Asumimos 'REAL')
                                        if t.get('mode', 'REAL') == 'PAPER': 
                                            continue

                                        real_trades_count += 1
                                        monto = t.get('pnl_usd', 0.0)
                                        total_usd += monto
                                        
                                        # Fecha
                                        raw_time = t.get('time', '')
                                        if raw_time and ' ' in raw_time:
                                            try:
                                                if raw_time.split()[0] == today_str:
                                                    diario_usd += monto
                                            except: pass
                                        
                                        if t.get('resultado') == 'WIN': wins += 1

                                    # WinRate (Sobre trades reales)
                                    if real_trades_count > 0:
                                        wr = (wins / real_trades_count) * 100
                                    else:
                                        wr = 0.0
                                    
                                    icon_total = "🤑" if total_usd >= 0 else "🔻"
                                    icon_daily = "🌤️" if diario_usd >= 0 else "🌧️"
                                    
                                    msg = (f"💰 <b>PNL (DINERO REAL)</b>\n\n"
                                           f"{icon_daily} <b>Hoy:</b> ${diario_usd:.2f}\n"
                                           f"{icon_total} <b>Total:</b> ${total_usd:.2f}\n\n"
                                           f"📊 <b>Estadísticas Reales:</b>\n"
                                           f"Trades: {real_trades_count}\n"
                                           f"Win Rate: {wr:.1f}%")
                                    
                                    if len(historial) > real_trades_count:
                                        msg += f"\n\n<i>(Se ocultaron {len(historial)-real_trades_count} trades de Paper)</i>"
                                    
                                    enviar_telegram(msg)

                                except Exception as e:
                                    enviar_telegram(f"❌ Error PnL: {e}")

                            # --- INFO (CONFIANZA + UMBRALES) ---
                            elif texto.startswith('/info'):
                                try:
                                    parts = texto.split()
                                    target_sym = parts[1].upper() if len(parts) > 1 else "BTC/USDT"
                                    
                                    # --- 1. DATOS CRUDOS ---
                                    raw_cluster = bot_state.get("market_cluster", 0)
                                    reglas_hmm = REGLAS_HMM.get(raw_cluster, [])
                                    ai_data = bot_state.get("ai_views", {}).get(target_sym)

                                    # Diccionario para mapear texto a índice
                                    map_reg = {'RAN':0, 'BULL':1, 'BEAR':2, 'CAOS':3}

                                    
                                    m_str="N/A"; m_conf=0.0; m_idx=0
                                    t_str="N/A"; t_conf=0.0; t_idx=0
                                    
                                    if ai_data:
                                        # Parseamos Macro
                                        try:
                                            m_raw = ai_data['MACRO'].split(' (')
                                            m_str = m_raw[0]
                                            m_conf = float(m_raw[1].replace(')', ''))
                                            m_idx = map_reg.get(m_str, 0)
                                        except: pass

                                        # Parseo Táctico
                                        try:
                                            t_raw = ai_data['TACTICO'].split(' (')
                                            t_str = t_raw[0]
                                            t_conf = float(t_raw[1].replace(')', ''))
                                            t_idx = map_reg.get(t_str, 0)
                                        except: pass

                                    # --- 2. CÁLCULO DE LÓGICA ESTÁNDAR (SIMULACIÓN) ---
                                    # Obtenemos los umbrales dinámicos actuales para comparar
                                    # Usamos 'BUY' como referencia base, aunque puede variar levemente para SELL
                                    ref_ctx, ref_tac = obtener_umbrales_ajustados('BUY') 
                                    
                                    # A. MACRO
                                    pass_macro = m_conf >= ref_ctx
                                    
                                    # B. TÁCTICO
                                    # Cada régimen táctico tiene su propio umbral
                                    umb_tac_req = ref_tac.get(t_idx, 0.30)
                                    pass_tactico = t_conf >= umb_tac_req

                                    # C. Determinación de Estrategias Estándar Permitidas
                                    # Lógica del main_loop
                                    std_ops = [] # Lista de tuplas (Estrategia, Dirección)
                                    
                                    if pass_macro and pass_tactico:
                                        # Lógica de Regímenes Tácticos
                                        if t_idx == 0: # Rango
                                            std_ops.append(('RANGO', 'BOTH'))
                                            if m_idx != 0: std_ops.append(('FRPV', 'BOTH'))
                                            
                                        elif t_idx == 1: # Bull
                                            if m_idx != 2: # Si Macro no es Bear
                                                for s in ['FRPV', 'TREND', 'BREAKOUT', 'RANGO']: std_ops.append((s, 'LONG'))
                                        
                                        elif t_idx == 2: # Bear
                                            if m_idx != 1: # Si Macro no es Bull
                                                for s in ['FRPV', 'TREND', 'BREAKOUT', 'RANGO']: std_ops.append((s, 'SHORT'))
                                            
                                        elif t_idx == 3: # Caos
                                            if m_idx == 1: 
                                                std_ops.append(('BREAKOUT', 'LONG')); std_ops.append(('FRPV', 'LONG'))
                                            elif m_idx == 2: 
                                                std_ops.append(('BREAKOUT', 'SHORT')); std_ops.append(('FRPV', 'SHORT'))
                                            elif m_idx == 0:
                                                std_ops.append(('FRPV', 'BOTH'))

                                    # --- 3. MENSAJE ---
                                    msg = f"🔍 <b>ANÁLISIS: {target_sym}</b>\n"
                                    msg += f"🌎 Contexto: C{raw_cluster}\n"
                                    
                                    
                                    icon_m = "✅" if pass_macro else f"⛔(Req {ref_ctx:.2f})"
                                    icon_t = "✅" if pass_tactico else f"⛔(Req {umb_tac_req:.2f})"
                                    
                                    msg += f"🧠 Jefe: {m_str} ({m_conf:.2f}) {icon_m}\n"
                                    msg += f"⚔️ Soldado: {t_str} ({t_conf:.2f}) {icon_t}\n\n"
                                    
                                    # --- SECCIÓN A: HMM (VIP) ---
                                    msg += "👑 <b>HMM (Reglas VIP):</b>\n"
                                    if not reglas_hmm: msg += " • (Sin reglas activas)\n"
                                    else:
                                        for regla in reglas_hmm:
                                            base_strat = regla.split('_')[0]
                                            key = f"{target_sym}_{base_strat}"
                                            st = bot_state.get("strategy_state", {}).get(key, {})
                                            
                                            lev = st.get("leverage", LEVERAGE_BASE)
                                            
                                            cool_u = st.get("cooldown_until")
                                            if cool_u and datetime.now() < datetime.fromisoformat(cool_u): status="❄️"
                                            else: status="✅"
                                            
                                            nombre = regla.replace('_', ' ')
                                            msg += f" • {nombre}: {status} x{lev}\n"
                                    msg += "\n"

                                    # --- SECCIÓN B: CEREBRO ESTÁNDAR ---
                                    msg += "🧠 <b>ESTÁNDAR (Según Confianza):</b>\n"
                                    
                                    if not std_ops:
                                        if not pass_macro: msg += "⛔ BLOQUEADO POR JEFE (Baja Conf)\n"
                                        elif not pass_tactico: msg += "⛔ BLOQUEADO POR SOLDADO (Baja Conf)\n"
                                        else: msg += "⛔ SIN ESTRATEGIAS COMPATIBLES\n"
                                    else:
                                        
                                        strats_std_map = {}
                                        for s, d in std_ops: strats_std_map[s] = d
                                        
                                        for strat in CONFIG_STRAT.keys():
                                            if strat in strats_std_map:
                                                direccion = strats_std_map[strat]
                                                dir_icon = "L" if direccion == 'LONG' else "S" if direccion == 'SHORT' else "L/S"
                                                
                                                key = f"{target_sym}_{strat}"
                                                st = bot_state.get("strategy_state", {}).get(key, {})
                                                lev = st.get("leverage", LEVERAGE_BASE)
                                                
                                                # Check VIP Racha
                                                vip_u = st.get("vip_until")
                                                es_vip = vip_u and datetime.now() < datetime.fromisoformat(vip_u)
                                                icon_tipo = "👑" if es_vip else "⚡"
                                                
                                                # Check Cooldown
                                                cool_u = st.get("cooldown_until")
                                                is_cool = cool_u and datetime.now() < datetime.fromisoformat(cool_u)
                                                stat = "❄️" if is_cool else "✅"
                                                
                                                msg += f" • {strat} ({dir_icon}): {icon_tipo}{stat} x{lev}\n"

                                    enviar_telegram(msg)
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Info: {e}")

                            # --- GERENTE ---
                            elif texto in ['/gerente', '/manager', 'gerente']:
                                try:
                                    if not umbrales_pro:
                                        msg = "👔 <b>GERENTE V2:</b> 🔴 No cargado (o vacío)."
                                    else:
                                        msg = "👔 <b>POLÍTICAS DEL GERENTE (V2)</b>\n"
                                        msg += "<i>Umbrales de Probabilidad para intervenir</i>\n\n"
                                        
                                        for strat, sides in umbrales_pro.items():
                                            msg += f"🔹 <b>{strat}</b>\n"
                                            for side, rules in sides.items():
                                                veto = rules.get('veto', 0)
                                                agg = rules.get('agresivo', 0.99)
                                                
                                                
                                                s_icon = "⬆️" if side == 'BUY' else "⬇️"
                                                agg_txt = f"{agg:.2f}" if agg < 0.90 else "OFF"
                                                
                                                msg += f"   {s_icon} {side}: 🛡️&lt;{veto:.2f}  🔥&gt;{agg_txt}\n"
                                            msg += "\n"
                                        
                                        msg += "🛡️ = Veto (Baja a x1)\n🔥 = Sniper (Sube a x12)"
                                    
                                    enviar_telegram(msg)
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Gerente: {e}")

                            # --- SESGO FULL DATA (HISTORIAL + UMBRALES + ESTRATEGIA) ---
                            elif texto in ['/sesgo', '/bias', '/winrate']:
                                try:
                                    # 1. ANÁLISIS HISTÓRICO GLOBAL (Últimos 40 trades)
                                    hist = bot_state.get("closed_history", [])[-40:]
                                    
                                    longs = [x for x in hist if x['lado']=='BUY']
                                    shorts = [x for x in hist if x['lado']=='SELL']
                                    
                                    if longs:
                                        wins_l = sum(1 for x in longs if x['resultado']=='WIN')
                                        wr_l = (wins_l / len(longs)) * 100
                                    else: wr_l = 0.0
                                    
                                    if shorts:
                                        wins_s = sum(1 for x in shorts if x['resultado']=='WIN')
                                        wr_s = (wins_s / len(shorts)) * 100
                                    else: wr_s = 0.0
                                    
                                    # 2. CONSULTAR VARA DE MEDIR ACTUAL (AJUSTE DINÁMICO GLOBAL)
                                    ctx_l, tac_l_dict = obtener_umbrales_ajustados('BUY')
                                    ctx_s, tac_s_dict = obtener_umbrales_ajustados('SELL')
                                    
                                    base_tac_bull = UMBRALES_TACTICOS[1]
                                    base_tac_bear = UMBRALES_TACTICOS[2]
                                    
                                    msg = "⚖️ <b>BALANZA DE PODER (V4.3)</b>\n\n"
                                    
                                    msg += "📊 <b>Global (Last 40):</b>\n"
                                    msg += f"   🟢 Longs:    {wr_l:.0f}%    ({len(longs)} ops)\n"
                                    msg += f"   🔴 Shorts: {wr_s:.0f}%    ({len(shorts)} ops)\n\n"
                                    
                                    msg += "🎯 <b>Umbrales Globales (Jefe/Soldado):</b>\n"
                                    diff_l = UMBRAL_CONTEXTO - ctx_l
                                    estado_l = "🔥Easy" if diff_l > 0 else "🛡️Hard" if diff_l < 0 else "Neutro"
                                    msg += f"⬆️ <b>LONG ({estado_l}):</b> {ctx_l:.2f} / {tac_l_dict[1]:.2f}\n"
                                    
                                    diff_s = UMBRAL_CONTEXTO - ctx_s
                                    estado_s = "🔥Easy" if diff_s > 0 else "🛡️Hard" if diff_s < 0 else "Neutro"
                                    msg += f"⬇️ <b>SHORT ({estado_s}):</b> {ctx_s:.2f} / {tac_s_dict[2]:.2f}\n\n"
                                    
                                    # --- 3. AJUSTE FINO POR ESTRATEGIA (PnL Reciente) ---
                                    msg += "🎛️ <b>AJUSTE POR ESTRATEGIA:</b>\n"
                                    msg += "<i>(🟢Bonus = Facilita | 🛡️Castigo = Endurece)</i>\n"
                                    
                                    for s_name in CONFIG_STRAT.keys():
                                        # Calculamos el sesgo para Long y Short de esta estrategia
                                        adj_l = calcular_sesgo_estrategia(s_name, 'BUY')
                                        adj_s = calcular_sesgo_estrategia(s_name, 'SELL')
                                        
                                        # Long
                                        if adj_l < 0: txt_l = f"🟢{adj_l:.2f}" # Resta al umbral (facilita)
                                        elif adj_l > 0: txt_l = f"🛡️+{adj_l:.2f}" # Suma al umbral (endurece)
                                        else: txt_l = "⚪0.00"
                                        
                                        # Short
                                        if adj_s < 0: txt_s = f"🟢{adj_s:.2f}"
                                        elif adj_s > 0: txt_s = f"🛡️+{adj_s:.2f}"
                                        else: txt_s = "⚪0.00"
                                        
                                        msg += f"🔹 <b>{s_name}:</b> L:{txt_l} | S:{txt_s}\n"
                                    
                                    enviar_telegram(msg)
                                        
                                except Exception as e:
                                    enviar_telegram(f"❌ Error calculando sesgo: {e}")

                            # --- AUDIT (VER CASTIGOS) ---
                            elif texto in ['/audit', 'castigos', 'audit']:
                                msg = "👮 <b>AUDITORÍA DE ESTRATEGIAS</b>\n(Solo mostrando anomalías)\n\n"
                                found = False
                                for key, st in bot_state.get("strategy_state", {}).items():
                                    lev = st.get("leverage", LEVERAGE_BASE)
                                    status = st.get("status", "NORMAL")
                                    cool_u = st.get("cooldown_until")
                                    en_cool = cool_u and datetime.now() < datetime.fromisoformat(cool_u)
                                    
                                    # (leverage cambiado, status penalty, o cooldown)
                                    if lev != LEVERAGE_BASE or status != "NORMAL" or en_cool:
                                        found = True
                                        sym = key.split('_')[0]
                                        strat = key.split('_')[-1]
                                        
                                        msg += f"🔸 <b>{sym} {strat}</b>\n"
                                        msg += f"   Estado: {status}\n"
                                        msg += f"   Lev: x{lev}\n"
                                        if en_cool: msg += f"   ❄️ Cooldown: {cool_u.split(' ')[1][:5]}\n"
                                        msg += "\n"
                                
                                if not found: msg += "✅ Todo opera bajo parámetros normales."
                                enviar_telegram(msg)

                            # --- VER POSICIONES ---
                            elif texto in ['/trades', 'posiciones', 'cartera']:
                                activos = bot_state.get("active_trades", {})
                                if not activos:
                                    enviar_telegram("📭 <b>No hay operaciones abiertas.</b>")
                                else:
                                    msg = "💼 <b>CARTERA ACTIVA:</b>\n\n"
                                    try:
                                        
                                        unique_syms = list(set([v['symbol'] for v in activos.values()]))
                                        tickers = {}
                                        
                                        
                                        try: 
                                            raw_tickers = exchange.fetch_tickers(unique_syms)
                                            tickers = raw_tickers
                                        except: pass 

                                        for trade_id, data in activos.items():
                                            sym = data['symbol']
                                            side = data['side']
                                            strat = data['strategy']
                                            entry = float(data['price'])
                                            margin = float(data.get('margin_used', 0))
                                            lev = int(data.get('leverage', 1))
                                            
                                            # Intentamos buscar el precio actual de varias formas
                                            curr = None
                                            if sym in tickers: curr = tickers[sym]['last']
                                            
                                            elif f"{sym}:USDT" in tickers: curr = tickers[f"{sym}:USDT"]['last']
                                            
                                            
                                            if curr is None:
                                                try: curr = exchange.fetch_ticker(sym)['last']
                                                except: pass

                                            pnl_txt = ""
                                            if curr:
                                                mult = 1 if side == 'BUY' else -1
                                                pnl_pct = ((curr - entry) / entry) * mult * lev * 100
                                                pnl_usd = (margin * pnl_pct) / 100
                                                icon = "🟢" if pnl_pct > 0 else "🔴"
                                                pnl_txt = f"\n   PnL: {icon} {pnl_pct:.2f}% (${pnl_usd:.2f})"

                                            msg += (f"🔹 <b>{sym}</b> {side} ({strat})\n"
                                                    f"   x{lev} | Entry: {entry}"
                                                    f"{pnl_txt}\n"
                                                    f"   Margen: ${margin:.1f}\n"
                                                    f"   {data.get('meta_info', '')}\n\n") 
                                        enviar_telegram(msg)
                                    except Exception as e:
                                        enviar_telegram(f"⚠️ Error listando trades: {e}")
                            
                            elif texto in ['/balance', 'saldo', 'wallet']:
                                try:
                                    # Saldo USDT disponible y total en Futuros
                                    bal = exchange.fetch_balance()['USDT']
                                    disponible = bal['free']
                                    total = bal['total']
                                    msg = f"🏦 <b>BILLETERA BINANCE</b>\n\n💵 Total: ${total:.2f}\n🔓 Disponible: ${disponible:.2f}\n⚙️ Configurado en Bot: ${CAPITAL_MAXIMO:.2f}"
                                    enviar_telegram(msg)
                                except Exception as e:
                                    enviar_telegram(f"⚠️ Error al leer balance: {e}")

                            
                            #  HISTORIAL FINANCIERO
                            
                            elif texto.startswith('/historial'):
                                try:
                                    # 1. Obtener cantidad solicitada (Default 10)
                                    parts = texto.split()
                                    limit = int(parts[1]) if len(parts) > 1 else 10
                                    
                                    # 2. Leer de la MEMORIA ESTRUCTURADA (No del txt)
                                    
                                    hist_data = bot_state.get("closed_history", [])
                                    
                                    if not hist_data:
                                        enviar_telegram("📭 <b>Tu historial está vacío.</b>\nEl bot aún no ha cerrado operaciones.")
                                    else:
                                        # Lo más reciente primero
                                        recent = hist_data[-limit:][::-1]
                                        
                                        msg = f"🗄️ <b>HISTORIAL CERRADO (Últimos {len(recent)})</b>\n\n"
                                        
                                        for t in recent:
                                            # Extraer datos del diccionario
                                            strat = t.get('strat', 'UNK')
                                            side = "LONG" if t.get('lado') == 'BUY' else "SHORT"
                                            res = t.get('resultado', 'NEUTRAL')
                                            pnl_usd = t.get('pnl_usd', 0.0)
                                            pnl_pct = t.get('pnl_pct', 0.0)
                                            mode = t.get('mode', 'REAL')
                                            
                                            # Formateo Visual
                                            if res == 'WIN': icon = "🟢"
                                            elif res == 'LOSS': icon = "🔴"
                                            else: icon = "🛡️"
                                            
                                            # Solo mostramos la hora
                                            try: hora = t['time'].split()[1][:5]
                                            except: hora = "--:--"
                                            
                                            tag_paper = "[P]" if mode == 'PAPER' else "[R]"
                                            
                                            msg += f"{icon} <b>{strat}</b> {side} {tag_paper}\n"
                                            msg += f"   {hora} | {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n"
                                        
                                        enviar_telegram(msg)

                                except Exception as e:
                                    enviar_telegram(f"❌ Error historial: {e}")
                            
                            elif texto in ['/sleep', 'sleep']:
                                bot_paused = True
                                enviar_telegram("💤 <b>MODO SLEEP ACTIVADO</b>\nEl bot NO abrirá nuevas posiciones.")
                            
                            elif texto in ['/wake', 'wake', 'despertar']:
                                bot_paused = False
                                enviar_telegram("⚡ <b>MODO ACTIVO</b>\nBuscando oportunidades nuevamente.")

                            elif texto in ['/cerrar', 'cerrar', 'panic']:
                                enviar_telegram("🚨 <b>RECIBIDO: CERRAR TODO</b>\nEjecutando pánico...")
                                cerrar_todas_posiciones(exchange)

                            
                            # SILENCIAR AUTO-SYNC
                            
                            elif texto in ['/mutesync', '/silenciosync', 'mute']:
                                SILENCIAR_AUTO_SYNC = not SILENCIAR_AUTO_SYNC
                                
                                if SILENCIAR_AUTO_SYNC:
                                    msg = "🔕 <b>AUTO-SYNC SILENCIADO</b>\nEl bot seguirá corrigiendo errores en silencio, pero no te enviará alertas."
                                else:
                                    msg = "🔔 <b>AUTO-SYNC ACTIVADO</b>\nTe avisaré si detecto fantasmas, aliens o conflictos."
                                
                                enviar_telegram(msg)
                            
                            
                            # COMANDO 1: REINICIAR (Simulamos un error)
                            
                            elif texto in ['/reboot', '/reiniciar', 'reboot']:
                                enviar_telegram("🔄 <b>REINICIANDO SISTEMA...</b>\nLimpiando RAM y recargando código.\nVuelvo en 5 segundos... ♻️")
                                print("🔄 REINICIO SOLICITADO: Saliendo con código 1 (Error simulado).")
                                
                                time.sleep(3)  # Damos tiempo a enviar el mensaje
                                os._exit(1)    # <--- Docker piensa que falló y reinicia.

                            
                            # APAGAR REAL (Salida limpia)
                            
                            elif texto in ['/stop', '/apagar', 'stop']:
                                enviar_telegram("🛑 <b>APAGANDO DEFINITIVAMENTE</b>\nEl contenedor se detendrá.\nPara prenderlo de nuevo, usa la consola de Vultr.")
                                print("🛑 APAGADO SOLICITADO: Saliendo con código 0 (Éxito).")
                                
                                time.sleep(3)  # Damos tiempo a enviar el mensaje
                                os._exit(0)    # <--- Docker ve que terminamos bien y no reinicia.
                                
                            # --- CAMBIAR CAPITAL ---
                            elif texto.startswith('/setcap'):
                                try:
                                    partes = texto.split()
                                    nuevo_cap = float(partes[1])
                                    
                                    # Actualizamos la variable global
                                    CAPITAL_MAXIMO = nuevo_cap
                                    
                                    # Calculamos cuánto es la nueva base
                                    base_usd = CAPITAL_MAXIMO * PORCENTAJE_BASE
                                    
                                    msg = (
                                        f"✅ <b>CAPITAL ACTUALIZADO</b>\n"
                                        f"💰 Nuevo Techo: ${CAPITAL_MAXIMO:.2f}\n"
                                        f"🌱 Nueva Base:  ${base_usd:.2f} ({int(PORCENTAJE_BASE*100)}%)\n\n"
                                        f"<i>El sistema 'Picante' pivotará sobre los ${base_usd:.2f}.</i>"
                                    )
                                    enviar_telegram(msg)
                                    print(f"💰 CAPITAL: ${CAPITAL_MAXIMO} | BASE: ${base_usd}")
                                    
                                except:
                                    enviar_telegram("⚠️ Error. Usa: /setcap 30")

                            # --- CAMBIAR MODO (REAL / PAPER) ---
                            elif texto.startswith('/mode') or texto.startswith('/modo'):
                                # Verificación de seguridad: ¿Hay trades abiertos?
                                if len(bot_state["active_trades"]) > 0:
                                    enviar_telegram("⚠️ <b>DENEGADO</b>\nNo puedes cambiar de modo con operaciones abiertas.\nUsa /cerrar primero o espera a que terminen.")
                                else:
                                    try:
                                        arg = texto.split()[1].upper()
                                        if arg in ['REAL', 'LIVE']:
                                            PAPER_TRADING = False
                                            enviar_telegram("💸 <b>MODO REAL ACTIVADO</b>\nCuidado: El bot ahora opera con dinero real.")
                                            print("\n⚠️ CAMBIO DE MODO: AHORA EN REAL MONEY ⚠️\n")
                                        elif arg in ['PAPER', 'DEMO', 'TEST']:
                                            PAPER_TRADING = True
                                            enviar_telegram("📄 <b>MODO PAPER ACTIVADO</b>\nOperaciones simuladas.")
                                            print("\n📄 CAMBIO DE MODO: AHORA EN PAPER TRADING\n")
                                        else:
                                            enviar_telegram("❌ Uso: <code>/mode REAL</code> o <code>/mode PAPER</code>")
                                    except:
                                        enviar_telegram("❌ Uso: <code>/mode REAL</code> o <code>/mode PAPER</code>")

                            # --- GESTIÓN MANUAL ---
                            elif texto.startswith('/mod'):
                                try:
                                    parts = texto.split()
                                    if len(parts) < 4: raise ValueError("Faltan argumentos")
                                    
                                    symbol = parts[1].upper()
                                    strat = parts[2].upper()
                                    accion = parts[3].upper() # VIP, UNVIP, BLOCK, RESET
                                    
                                    msg_res = ""
                                    
                                    # --- CASO A: MODIFICACIÓN GLOBAL ---
                                    if symbol in ['ALL', 'GLOBAL', 'TODO']:
                                        inicializar_stats_globales(strat)
                                        
                                        if accion == 'BLOCK':
                                            mins = int(parts[4]) if len(parts) > 4 else 60
                                            t_block = datetime.now() + timedelta(minutes=mins)
                                            bot_state["global_strat_perf"][strat]["cooldown_until"] = str(t_block)
                                            msg_res = f"⛔ <b>BLOQUEO GLOBAL: {strat}</b>\nDetenida por {mins} min."
                                            
                                        elif accion == 'VIP':
                                            t_vip = "2099-12-31 23:59:59"
                                            for k in bot_state["strategy_state"]:
                                                if k.endswith(f"_{strat}"):
                                                    bot_state["strategy_state"][k]["vip_until"] = t_vip
                                                    bot_state["strategy_state"][k]["status"] = "NORMAL"
                                            msg_res = f"👑 <b>VIP GLOBAL: {strat}</b>\nAplicado a todo el mercado."
                                            
                                        elif accion == 'UNVIP':
                                            for k in bot_state["strategy_state"]:
                                                if k.endswith(f"_{strat}"):
                                                    bot_state["strategy_state"][k]["vip_until"] = None
                                            msg_res = f"😐 <b>UNVIP GLOBAL: {strat}</b>\nAhora son mortales (mantienen racha)."

                                        elif accion == 'RESET':
                                            bot_state["global_strat_perf"][strat]["cooldown_until"] = None
                                            bot_state["global_strat_perf"][strat]["fails"] = 0
                                            for k in bot_state["strategy_state"]:
                                                if k.endswith(f"_{strat}"):
                                                    bot_state["strategy_state"][k]["vip_until"] = None
                                                    bot_state["strategy_state"][k]["cooldown_until"] = None
                                                    bot_state["strategy_state"][k]["status"] = "NORMAL"
                                                    bot_state["strategy_state"][k]["consecutive_losses"] = 0 
                                            msg_res = f"🔄 <b>RESET GLOBAL: {strat}</b>\nTodo limpio (Amnistía)."
                                        
                                    # --- CASO B: MODIFICACIÓN INDIVIDUAL ---
                                    else:
                                        key = f"{symbol}_{strat}"
                                        inicializar_estado_estrategia(key)
                                        st = bot_state["strategy_state"][key]

                                        if accion == 'VIP':
                                            st["vip_until"] = "2099-12-31 23:59:59"
                                            st["cooldown_until"] = None
                                            st["status"] = "NORMAL"
                                            msg_res = f"👑 <b>{key} ES VIP</b> (Indefinido)"
                                            
                                        elif accion == 'UNVIP':
                                            st["vip_until"] = None
                                            msg_res = f"😐 <b>{key} YA NO ES VIP</b>"

                                        elif accion == 'BLOCK':
                                            mins = int(parts[4]) if len(parts) > 4 else 60
                                            tiempo_bloqueo = datetime.now() + timedelta(minutes=mins)
                                            st["cooldown_until"] = str(tiempo_bloqueo)
                                            st["vip_until"] = None
                                            msg_res = f"⛔ <b>{key} BLOQUEADA</b>\nHasta: {tiempo_bloqueo.strftime('%H:%M')}"
                                            
                                        elif accion == 'RESET':
                                            st["vip_until"] = None
                                            st["cooldown_until"] = None
                                            st["status"] = "NORMAL"
                                            st["consecutive_losses"] = 0 
                                            msg_res = f"🔄 <b>{key} RESETEADA</b>\nEstado limpio."
                                        
                                        bot_state["strategy_state"][key] = st

                                    guardar_estado()
                                    enviar_telegram(msg_res)
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error. Ej: <code>/mod BTC/USDT TREND UNVIP</code>")

                            
                            # RECARGA BNB
                            
                            elif texto.startswith('/buybnb') or texto.startswith('/gas'):
                                
                                try:
                                    try: amount_usdt = float(texto.split()[1])
                                    except: amount_usdt = 10.0 
                                    
                                    enviar_telegram(f"⛽ <b>INICIANDO RECARGA...</b>\nAnalizando límites de Binance para comprar ${amount_usdt}...")

                                    # 1. INSTANCIA SPOT TEMPORAL
                                    exchange_spot = ccxt.binance({
                                        'apiKey': API_KEY, 
                                        'secret': API_SECRET, 
                                        'options': {'defaultType': 'spot'}
                                    })
                                    
                                    # 2. CARGAR REGLAS DE MERCADO (Para saber el mínimo)
                                    exchange_spot.load_markets()
                                    market_bnb = exchange_spot.market('BNB/USDT')
                                    min_cost = market_bnb['limits']['cost']['min'] 
                                    
                                    # --- VALIDACIÓN DE MONTO ---
                                    if amount_usdt < min_cost:
                                        raise ValueError(f"⚠️ <b>Monto muy bajo.</b>\nBinance exige un mínimo de <b>${min_cost} USDT</b> para operar BNB.\nIntenta con: <code>/buybnb {min_cost + 1}</code>")

                                    # 3. VERIFICAR SALDO EN SPOT
                                    bal_spot = exchange_spot.fetch_balance()['USDT']['free']
                                    if bal_spot < amount_usdt:
                                        raise ValueError(f"💸 <b>Falta saldo en SPOT.</b>\nTienes: ${bal_spot:.2f}\nNecesitas: ${amount_usdt:.2f}\n\nUsa <code>/bank {amount_usdt} FUT SPOT</code> para traer dinero del Bot.")

                                    # 4. COMPRAR BNB (MARKET)
                                    ticker = exchange_spot.fetch_ticker('BNB/USDT')
                                    price = ticker['last']
                                    amount_bnb = amount_usdt / price
                                    amt_final = exchange_spot.amount_to_precision('BNB/USDT', amount_bnb)
                                    
                                    enviar_telegram(f"🛒 Comprando {amt_final} BNB a precio market...")
                                    order = exchange_spot.create_order('BNB/USDT', 'market', 'buy', amt_final)
                                    
                                    # Calcular lo real comprado (restando fee de la compra)
                                    bnb_bought = float(order['amount']) 
                                    if 'fee' in order and order['fee']:
                                        bnb_bought -= float(order['fee']['cost'])
                                    
                                    enviar_telegram(f"✅ <b>COMPRA SPOT EXITOSA</b>\nSe obtuvieron: {bnb_bought:.4f} BNB")
                                    
                                    # 5. TRANSFERIR A FUTUROS
                                    enviar_telegram("🔄 <b>ENVIANDO A FUTUROS...</b>")
                                    exchange.transfer('BNB', bnb_bought, 'spot', 'future')
                                    
                                    enviar_telegram(f"⛽ <b>¡TANQUE LLENO!</b>\n{bnb_bought:.4f} BNB listos para comisiones.")
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Recarga: {e}")

                            
                            # OPERAR EN SPOT
                            
                            elif texto.startswith('/spot'):
                                
                                try:
                                    parts = texto.split()
                                    if len(parts) < 4: raise ValueError("Faltan datos")
                                    
                                    raw_sym = parts[1].upper()
                                    symbol = f"{raw_sym}/USDT" if "/" not in raw_sym else raw_sym
                                    side = parts[2].lower() # buy / sell
                                    usdt_amount = float(parts[3])
                                    
                                    enviar_telegram(f"🛒 <b>OPERACIÓN SPOT ({side.upper()})</b>\n{symbol} por ${usdt_amount}")

                                    # 1. Instancia Spot
                                    exchange_spot = ccxt.binance({
                                        'apiKey': API_KEY, 'secret': API_SECRET, 
                                        'options': {'defaultType': 'spot'}
                                    })
                                    
                                    # 2. Cálculos
                                    ticker = exchange_spot.fetch_ticker(symbol)
                                    price = ticker['last']
                                    
                                    if side == 'buy':
                                        qty = usdt_amount / price
                                    else:
                                    
                                        # El input es Valor en USDT.
                                        qty = usdt_amount / price
                                    
                                    qty_final = exchange_spot.amount_to_precision(symbol, qty)
                                    
                                    # 3. Ejecutar
                                    order = exchange_spot.create_order(symbol, 'market', side, qty_final)
                                    
                                    enviar_telegram(f"✅ <b>SPOT EJECUTADO</b>\nPrecio: {order['average']}\nQty: {order['amount']}")
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Spot: {e}\nUso: <code>/spot BTC BUY 50</code>")

			                
                            # OPERATIVA MANUAL (SIEMPRE REAL)
                            
                            elif texto.startswith('/operar') or texto.startswith('/trade'):
                                # Sintaxis: /operar [MONEDA] [LADO] [LEV] [MARGEN] [TP] [SL]
                                try:
                                    parts = texto.split()
                                    if len(parts) < 7:
                                        raise ValueError("Faltan datos.")

                                    # 1. Parsing de argumentos
                                    raw_sym = parts[1].upper()
                                    symbol = f"{raw_sym}/USDT" if "/" not in raw_sym else raw_sym 
                                    side = parts[2].upper() 
                                    lev_manual = int(parts[3])
                                    margin_manual = float(parts[4])
                                    tp_manual = float(parts[5])
                                    sl_manual = float(parts[6])

                                    if side not in ['BUY', 'SELL']: raise ValueError("Lado incorrecto (BUY/SELL)")

                                    enviar_telegram(f"🫡 <b>PROCESANDO ORDEN MANUAL (REAL)...</b>\n{symbol} {side} x{lev_manual} (${margin_manual})")

                                    # 2. Validaciones de Mercado
                                    ticker = exchange.fetch_ticker(symbol)
                                    curr_price = ticker['last']
                                    market_info = exchange.market(symbol)
                                    
                                    # Datos mínimos del Exchange
                                    min_qty = market_info['limits']['amount']['min']
                                    min_cost = market_info['limits']['cost']['min'] if 'cost' in market_info['limits'] else 5.0
                                    
                                    
                                    notional_value = margin_manual * lev_manual
                                    amount_raw = notional_value / curr_price
                                    
                                    
                                    if amount_raw < min_qty:
                                        req_notional = min_qty * curr_price
                                        req_margin = req_notional / lev_manual
                                        raise ValueError(
                                            f"Monto insuficiente para Binance.\n"
                                            f"Min Qty: {min_qty} {raw_sym} (~${req_notional:.2f})\n"
                                            f"Tú enviaste: {amount_raw:.5f} {raw_sym} (${notional_value:.2f})\n"
                                            f"💡 Solución: Sube el margen a <b>${req_margin:.2f}</b> o aumenta el apalancamiento."
                                        )

                                    if notional_value < min_cost:
                                        raise ValueError(f"Valor total muy bajo (${notional_value}). Mínimo Binance: ${min_cost}")

                                    # 3. Ajuste de precisión
                                    amount_final = exchange.amount_to_precision(symbol, amount_raw)
                                    tp_final = exchange.price_to_precision(symbol, tp_manual)
                                    sl_final = exchange.price_to_precision(symbol, sl_manual)

                                    # 4. Ejecución forzada
                                    # ignoramos PAPER_TRADING. Si es manual ->  REAL.
                                    qty_lograda = float(amount_final)
                                    entry_real = curr_price
                                    
                                    try:
                                        # Configurar Margin/Leverage
                                        try: exchange.set_margin_mode('ISOLATED', symbol)
                                        except: pass
                                        exchange.set_leverage(lev_manual, symbol)

                                        # Orden de Mercado
                                        order = exchange.create_order(symbol, 'market', side.lower(), amount_final)
                                        qty_lograda = order['amount']
                                        entry_real = order['average'] if order['average'] else curr_price

                                        # Protecciones (SL y TP)
                                        side_exit = 'sell' if side == 'BUY' else 'buy'
                                        exchange.create_order(symbol, 'STOP_MARKET', side_exit, qty_lograda, None, {'stopPrice': sl_final})
                                        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', side_exit, qty_lograda, None, {'stopPrice': tp_final})
                                        
                                    except Exception as e:
                                        enviar_telegram(f"❌ <b>ERROR BINANCE:</b> {e}")
                                        raise e 

                                    # 5. REGISTRAR EN CEREBRO DEL BOT
                                    trade_id = f"{symbol}_MANUAL_{int(time.time())}" 
                                    
                                    with memory_lock: 
                                        bot_state["active_trades"][trade_id] = {
                                            'entry_time': str(datetime.now()),
                                            'price': entry_real,
                                            'tp': tp_manual,
                                            'sl': sl_manual,
                                            'side': side,
                                            'symbol': symbol,
                                            'strategy': 'MANUAL',
                                            'leverage': lev_manual,
                                            'margin_used': margin_manual,
                                            'meta_info': 'Operada desde Telegram (Proxy Real)',
                                            'amount': qty_lograda,
                                            'be_triggered': False,
                                            'mode': 'REAL' 
                                        }
                                    guardar_estado()

                                    msg_exito = (
                                        f"✅ <b>ORDEN MANUAL (REAL) EJECUTADA</b>\n"
                                        f"{symbol} {side} x{lev_manual}\n"
                                        f"Entry: {entry_real}\n"
                                        f"TP: {tp_final} | SL: {sl_final}\n"
                                        f"Margen: ${margin_manual}\n"
                                        f"<i>El bot la gestionará aunque esté en modo Paper.</i>"
                                    )
                                    enviar_telegram(msg_exito)

                                except ValueError as ve:
                                    enviar_telegram(f"⚠️ <b>Error Validación:</b>\n{ve}")
                                except Exception as e:
                                    print(f"❌ Error Manual: {e}")
                                    enviar_telegram(f"❌ Error ejecutando manual: {e}")

                            
                            # LIMIT SNIPER - CON VALIDACIÓN
                            
                            elif texto.startswith('/limit'):
                                # /limit BTC BUY 95000 20 100 96000 94000
                                try:
                                    parts = texto.split()
                                    if len(parts) < 8: 
                                        raise ValueError("Faltan datos. Sintaxis: /limit PAR SIDE PRECIO LEV MARGEN TP SL")

                                    raw_sym = parts[1].upper()
                                    symbol = f"{raw_sym}/USDT" if "/" not in raw_sym else raw_sym 
                                    side = parts[2].upper() 
                                    price_limit = float(parts[3])
                                    lev = int(parts[4])
                                    margin = float(parts[5])
                                    tp = float(parts[6])
                                    sl = float(parts[7])

                                    if side not in ['BUY', 'SELL']: raise ValueError("Lado debe ser BUY o SELL")
                                    
                                    # --- VALIDACIÓN LÓGICA DE PRECIOS (ERROR -2021) ---
                                    if side == 'BUY':
                                        if tp <= price_limit: raise ValueError(f"LONG ERROR: El TP ({tp}) debe ser MAYOR a la entrada ({price_limit})")
                                        if sl >= price_limit: raise ValueError(f"LONG ERROR: El SL ({sl}) debe ser MENOR a la entrada ({price_limit})")
                                    else: # SELL
                                        if tp >= price_limit: raise ValueError(f"SHORT ERROR: El TP ({tp}) debe ser MENOR a la entrada ({price_limit})")
                                        if sl <= price_limit: raise ValueError(f"SHORT ERROR: El SL ({sl}) debe ser MAYOR a la entrada ({price_limit})")
                                    # -------------------------------------------------------

                                    enviar_telegram(f"⏳ <b>COLOCANDO LIMIT...</b>\n{symbol} {side} @ ${price_limit}")

                                    # Calcular cantidad
                                    notional = margin * lev
                                    amount_raw = notional / price_limit
                                    amount = exchange.amount_to_precision(symbol, amount_raw)
                                    price_final = exchange.price_to_precision(symbol, price_limit)
                                    
                                    # Ejecutar Limit Order
                                    try:
                                        try: exchange.set_leverage(lev, symbol)
                                        except: pass
                                        
                                        # Limit Order "Desnuda"
                                        order = exchange.create_order(symbol, 'limit', side.lower(), amount, price_final)
                                        order_id = order['id']
                                        
                                        msg = (f"🔭 <b>VIGILANTE ACTIVADO ({symbol})</b>\n"
                                               f"Orden ID: <code>{order_id}</code>\n"
                                               f"Esperando precio: {price_final}...\n"
                                               f"<i>Cuando entre, pondré SL ({sl}) y TP ({tp}) automáticos.</i>")
                                        enviar_telegram(msg)
                                        
                                        # VIGILANTE
                                        t_vig = threading.Thread(target=monitor_limit_order, args=(exchange, symbol, order_id, side, lev, margin, tp, sl))
                                        t_vig.daemon = True 
                                        t_vig.start()
                                        
                                    except Exception as e:
                                        enviar_telegram(f"❌ Error Binance: {e}")

                                except Exception as e:
                                    enviar_telegram(f"❌ <b>RECHAZADO:</b> {e}")

                            
                            # MULTI-BILLETERA + FEES
                            
                            elif texto.startswith('/deposit') or texto.startswith('/direction'):
                                try:
                                    
                                    parts = texto.split()
                                    coin_target = parts[1].upper() if len(parts) > 1 else 'USDT'
                                    
                                    enviar_telegram(f"🕵️ <b>CONSULTANDO REDES Y COSTOS PARA {coin_target}...</b>")
                                    
                                    
                                    networks_target = {
                                        'TRX': 'TRC20 (Tron)',
                                        'BSC': 'BEP20 (BNB Smart Chain)',
                                        'SOL': 'SOL (Solana)',
                                        'ETH': 'ERC20 (Ethereum)',
                                        'SUI': 'SUI (Sui Network)',
                                        'MATIC': 'Polygon (Matic)',
                                        'AVAXC': 'AVAX C-Chain'
                                    }
                                    
                                    # Cargar datos estáticos del Exchange
                                    if not exchange.currencies:
                                        exchange.load_markets()
                                    
                                    currency_data = exchange.currencies.get(coin_target, {})
                                    networks_data = currency_data.get('networks', {})

                                    msg = f"📥 <b>DEPOSITAR {coin_target}</b>\n"
                                    msg += "<i>(Los precios mostrados son el costo de retiro estándar de la red)</i>\n\n"
                                    
                                    found_any = False

                                    # Iteramos cada red
                                    for net_code, net_name in networks_target.items():
                                        try:
                                            
                                            fee_val = None
                                            
                                            
                                            if net_code in networks_data:
                                                fee_val = networks_data[net_code].get('fee')
                                            
                                            
                                            if fee_val is not None:
                                                
                                                fee_txt = f"Costo Ref: <b>${fee_val}</b>"
                                                if fee_val > 5: fee_txt += " 🔴 (Caro)"
                                                elif fee_val < 1: fee_txt += " 🟢 (Barato)"
                                            else:
                                                fee_txt = "Costo Ref: (No disponible)"

                                            
                                            # Solo pedimos dirección si la red está habilitada para esa moneda
                                            can_deposit = True
                                            if net_code in networks_data:
                                                if not networks_data[net_code].get('deposit', True):
                                                    can_deposit = False
                                            
                                            if can_deposit:
                                                addr_info = exchange.fetch_deposit_address(coin_target, {'network': net_code})
                                                address = addr_info.get('address')
                                                tag = addr_info.get('tag')
                                                
                                                if address:
                                                    found_any = True
                                                    msg += f"🔸 <b>{net_name}</b>\n"
                                                    msg += f"   {fee_txt}\n"
                                                    msg += f"   <code>{address}</code>\n"
                                                    if tag: 
                                                        msg += f"   ⚠️ <b>MEMO/TAG:</b> <code>{tag}</code>\n"
                                                    msg += "\n"
                                        except:
                                            continue
                                    
                                    if found_any:
                                        msg += f"⚠️ <b>IMPORTANTE:</b> Envía SOLAMENTE <b>{coin_target}</b> por la red elegida. Si envías otra cosa, se pierde.\nUsa <code>/bank</code> cuando llegue."
                                        enviar_telegram(msg)
                                    else:
                                        enviar_telegram(f"❌ No encontré direcciones para {coin_target}.\n(Es posible que Binance tenga suspendidos los depósitos o no hayas generado la billetera en la web).")
                                        
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Deposit: {e}")

			                
                            # ESTADÍSTICAS POR ESTRATEGIA
                            
                            elif texto in ['/stats', 'estadisticas', 'performance']:
                                try:
                                    hist = bot_state.get("closed_history", [])
                                    if not hist:
                                        enviar_telegram("📭 <b>Sin historial suficiente.</b>")
                                        continue
                                    
                                    # Diccionario para agrupar
                                    stats = {} # {'FRPV': {'wins':0, 'total':0, 'pnl':0.0}, ...}
                                    
                                    for t in hist:
                                        s = t.get('strat', 'UNKNOWN')
                                        res = t.get('resultado', 'NEUTRAL')
                                        pnl = t.get('pnl_usd', 0.0)
                                        
                                        if s not in stats: stats[s] = {'wins': 0, 'total': 0, 'pnl': 0.0}
                                        
                                        stats[s]['total'] += 1
                                        stats[s]['pnl'] += pnl
                                        if res == 'WIN': stats[s]['wins'] += 1
                                    
                                    msg = "📊 <b>RENDIMIENTO POR ESTRATEGIA</b>\n\n"
                                    total_general = 0.0
                                    
                                    for strat, data in stats.items():
                                        wr = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
                                        icon = "🟢" if data['pnl'] > 0 else "🔴"
                                        msg += f"🔹 <b>{strat}</b>\n"
                                        msg += f"   Trades: {data['total']} | WR: {wr:.1f}%\n"
                                        msg += f"   PnL: {icon} ${data['pnl']:.2f}\n\n"
                                        total_general += data['pnl']
                                        
                                    msg += f"💰 <b>NETO TOTAL:</b> ${total_general:.2f}"
                                    enviar_telegram(msg)
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error stats: {e}")

                            
                            # LOGS DEL SISTEMA
                            
                            elif texto.startswith('/logs') or texto.startswith('/log'):
                                try:
                                    try: n_lines = int(texto.split()[1])
                                    except: n_lines = 15
                                    
                                    log_path = "live_trades_log.txt"
                                    if os.path.exists(log_path):
                                        with open(log_path, 'r', encoding='utf-8') as f:
                                            lines = f.readlines()
                                            
                                            
                                            lines = [l for l in lines if l.strip()]
                                            
                                            last_n = lines[-n_lines:]
                                            txt_log = "".join(last_n)
                                            
                                        if len(txt_log) > 4000: txt_log = txt_log[-4000:] 
                                        
                                        header = f"🛠️ <b>LOGS DEL SISTEMA (DEBUG)</b>\n<i>Eventos técnicos y decisiones internas:</i>\n\n"
                                        enviar_telegram(f"{header}<pre>{txt_log}</pre>")
                                    else:
                                        enviar_telegram("⚠️ El archivo de logs está vacío o no existe.")
                                except Exception as e:
                                    enviar_telegram(f"❌ Error leyendo logs: {e}")

                            
                            # REVELADOR DE NOMBRES (Para errores)
                            
                            elif texto.startswith('/nombres') or texto.startswith('/names'):
                                try:
                                    enviar_telegram("🕵️ <b>ESCANEO DE NOMBRES TÉCNICOS...</b>")
                                    
                                    # 1. Traemos posiciones crudas
                                    positions = exchange.fetch_positions()
                                    active = [p for p in positions if float(p['contracts']) > 0]
                                    
                                    if not active:
                                        enviar_telegram("📭 No hay posiciones abiertas para analizar.")
                                    else:
                                        msg = "🆔 <b>IDENTIDAD DE TUS POSICIONES:</b>\n\n"
                                        for p in active:
                                            
                                            nombre_ccxt = p['symbol'] 
                                            nombre_raw = p['info']['symbol']
                                            
                                            msg += f"🔹 <b>{nombre_ccxt}</b>\n"
                                            msg += f"   Bot lo ve como: '{nombre_ccxt}'\n"
                                            msg += f"   Binance lo llama: '{nombre_raw}'\n"
                                            msg += f"   -----------------\n"
                                        
                                        msg += "<i>Usa el nombre de 'Bot lo ve como' en tus comandos.</i>"
                                        enviar_telegram(msg)
                                        
                                except Exception as e:
                                    enviar_telegram(f"❌ Error: {e}")

                            
                            # DIAGNÓSTICO SEGURO (Para errores)
                            
                            elif texto.startswith('/dpos'):
                                try:
                                    
                                    parts = texto.split()
                                    if len(parts) < 2:
                                        enviar_telegram("⚠️ Uso: /dpos XRP")
                                        continue
                                        
                                    raw_sym = parts[1].upper()
                                    target = f"{raw_sym}/USDT"
                                    
                                    enviar_telegram(f"🧬 <b>ANALIZANDO {target}...</b>")
                                    
                                    # Buscamos la posición
                                    positions = exchange.fetch_positions([target])
                                    active = [p for p in positions if float(p['contracts']) > 0]
                                    
                                    if not active:
                                        enviar_telegram("❌ No hay posición activa en la API.")
                                    else:
                                        p = active[0]
                                        info = p.get('info', {})
                                        
                                        # ESCANEO DE CLAVES (Para encontrar el Leverage)
                                        
                                        candidates = [k for k in info.keys() if 'everage' in k.lower()]
                                        
                                        msg = f"🔍 <b>RESULTADOS PARA {raw_sym}:</b>\n\n"
                                        msg += f"✅ <b>POSICIÓN ENCONTRADA</b>\n"
                                        msg += f"Symbol CCXT: {p['symbol']}\n"
                                        msg += f"Symbol RAW: {info.get('symbol', 'N/A')}\n"
                                        msg += f"Margin Type: {info.get('marginType', 'N/A')}\n"
                                        msg += f"Position Side: {info.get('positionSide', 'N/A')}\n\n"
                                        
                                        msg += "🗝️ <b>CANDIDATOS A APALANCAMIENTO:</b>\n"
                                        for k in candidates:
                                            msg += f" • {k}: {info[k]}\n"
                                            
                                        if not candidates:
                                            msg += "⚠️ No encontré ninguna clave con nombre 'leverage'.\n"
                                            
                                        # 3. VERIFICACIÓN DE ÓRDENES
                                        msg += "\n📋 <b>TEST DE VISIBILIDAD DE ÓRDENES:</b>\n"
                                        
                                        # Prueba A: Nombre CCXT
                                        try:
                                            oa = exchange.fetch_open_orders(p['symbol'])
                                            msg += f"A) '{p['symbol']}': {len(oa)} órdenes\n"
                                        except: msg += f"A) '{p['symbol']}': Error\n"
                                        
                                        # Prueba B: Nombre RAW
                                        raw_name = info.get('symbol')
                                        if raw_name:
                                            try:
                                                ob = exchange.fetch_open_orders(raw_name)
                                                msg += f"B) '{raw_name}': {len(ob)} órdenes\n"
                                            except: msg += f"B) '{raw_name}': Error\n"
                                            
                                        enviar_telegram(msg)
                                        
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Fatal DPOS: {str(e)}")

                            
                            # EDITAR SL/TP
                            
                            elif texto.startswith('/edit') or texto.startswith('/modificar'):
                                # Uso: /edit ETH SL 3266
                                try:
                                    parts = texto.split()
                                    if len(parts) < 4: raise ValueError("Faltan datos (Uso: /edit XRP TP 2.6)")
                                    
                                    raw_sym = parts[1].upper()
                                    target_sym = f"{raw_sym}/USDT" if "/" not in raw_sym else raw_sym
                                    
                                    tipo = parts[2].upper() # SL o TP
                                    nuevo_precio = float(parts[3])
                                    
                                    enviar_telegram(f"📝 <b>EDITANDO {tipo} EN {target_sym}...</b>")

                                    # (Cantidad y Lado)
                                    qty = 0.0
                                    side_pos = None
                                    trade_id_found = None
                                    
                                    with memory_lock: 
                                        for tid, tdata in bot_state["active_trades"].items():
                                            if tdata['symbol'] == target_sym:
                                                qty = tdata['amount']; side_pos = tdata['side']; trade_id_found = tid
                                                break
                                    
                                    # Fallback a API si no está en memoria
                                    if qty == 0:
                                        try:
                                            pos = exchange.fetch_positions([target_sym])
                                            for p in pos:
                                                if float(p['contracts']) > 0:
                                                    qty = float(p['contracts'])
                                                    side_pos = 'BUY' if p['side'] == 'long' else 'SELL'
                                                    break
                                        except: pass
                                    
                                    if qty == 0:
                                        enviar_telegram(f"❌ No encontré posición activa en {target_sym}.")
                                        continue

                                    # OBTENER PRECIO ACTUAL (Para clasificar SL vs TP)
                                    try:
                                        curr_price = exchange.fetch_ticker(target_sym)['last']
                                    except:
                                        curr_price = nuevo_precio 

                                    # EJECUCIÓN (CREAR NUEVA)
                                    try:
                                        side_exit = 'sell' if side_pos == 'BUY' else 'buy'
                                        precio_fin = exchange.price_to_precision(target_sym, nuevo_precio)
                                        
                                        # Crear la nueva orden primero
                                        params = {'stopPrice': precio_fin, 'reduceOnly': True}
                                        if tipo == 'SL':
                                            ord_new = exchange.create_order(target_sym, 'STOP_MARKET', side_exit, qty, None, params)
                                        else:
                                            ord_new = exchange.create_order(target_sym, 'TAKE_PROFIT_MARKET', side_exit, qty, None, params)
                                        
                                        new_id = ord_new['id']
                                        
                                        # BORRAR VIEJAS
                                        candidatas = []
                                        try: 
                                            candidatas += exchange.fetch_open_orders(target_sym)
                                            candidatas += exchange.fetch_open_orders(target_sym, params={'stop': True})
                                        except: pass
                                        
                                        # ID
                                        candidatas_unicas = {o['id']: o for o in candidatas}.values()
                                        borradas = 0
                                        
                                        targets_del = [target_sym, f"{target_sym}:USDT", target_sym.replace('/', '')]

                                        for o in candidatas_unicas:
                                            if o['id'] == new_id: continue # No borra la nueva
                                            
                                            # Filtro de Lado (mismo lado que la nueva)
                                            if o['side'].lower() != side_exit.lower(): continue

                                            otype = str(o.get('type', '')).upper()
                                            trig = float(o.get('stopPrice') or o.get('triggerPrice') or 0)
                                            if trig == 0: continue 

                                            # --- (SL vs TP) ---
                                            o_role = 'UNKNOWN'
                                            
                                            # A) Por Etiqueta Explícita (Si existe)
                                            if 'STOP' in otype: o_role = 'SL'
                                            elif 'TAKE' in otype: o_role = 'TP'
                                            else:
                                                # B) Por Precio Relativo (Para órdenes 'COND')
                                                # Si estoy LONG: SL está abajo, TP arriba
                                                if side_pos == 'BUY': 
                                                    o_role = 'SL' if trig < curr_price else 'TP'
                                                # SHORT
                                                else: 
                                                    o_role = 'SL' if trig > curr_price else 'TP'
                                            
                                            # BORRAMOS (Si coincide rol)
                                            if tipo == o_role:
                                                orden_eliminada = False
                                                for t_name in targets_del:
                                                    try: exchange.cancel_order(o['id'], t_name); orden_eliminada = True
                                                    except: pass
                                                    if not orden_eliminada:
                                                        try: exchange.cancel_order(o['id'], t_name, params={'stop': True}); orden_eliminada = True
                                                        except: pass
                                                    if orden_eliminada:
                                                        borradas += 1
                                                        break

                                        enviar_telegram(f"✅ <b>CAMBIO EXITOSO</b>\nNuevo {tipo}: {precio_fin}\n🗑️ Viejas eliminadas: {borradas}")

                                        # C. ACTUALIZAR MEMORIA
                                        if trade_id_found:
                                            with memory_lock:
                                                if trade_id_found in bot_state["active_trades"]:
                                                    k_mem = 'sl' if tipo == 'SL' else 'tp'
                                                    bot_state["active_trades"][trade_id_found][k_mem] = float(nuevo_precio)
                                            guardar_estado()

                                    except Exception as e:
                                        enviar_telegram(f"❌ Error Binance: {e}")

                                except Exception as e:
                                    enviar_telegram(f"❌ Error Edit: {e}")
                                    
			                
                            # CERRAR SOBERANO (Cierra Real si existe / MATA ALIENS)
                            
                            elif texto.startswith('/close') or texto.startswith('/salir'):
                                try:
                                    parts = texto.split()
                                    if len(parts) < 2:
                                        enviar_telegram("⚠️ Uso: /close BTC")
                                        continue

                                    raw_sym = parts[1].upper()
                                    target_sym = f"{raw_sym}/USDT" if "/" not in raw_sym else raw_sym
                                    
                                    enviar_telegram(f"🫡 <b>BUSCANDO EN BINANCE {target_sym}...</b>")
                                    
                                    # 1. VERIFICACIÓN DE REALIDAD (BINANCE DIRECTO)
                                    
                                    pos_real_size = 0.0
                                    side_real = None
                                    
                                    try:
                                        positions = exchange.fetch_positions([target_sym])
                                        for p in positions:
                                            if float(p['contracts']) > 0:
                                                pos_real_size = float(p['contracts'])
                                                side_real = 'BUY' if p['side'] == 'long' else 'SELL'
                                                break
                                    except: pass

                                    # 2. EJECUCIÓN DE CIERRE REAL (Si existe en Binance)
                                    msg_binance = ""
                                    if pos_real_size > 0:
                                        try:
                                            # 1. Cancelar Todo
                                            try: exchange.cancel_all_orders(target_sym)
                                            except: pass
                                            try: exchange.cancel_all_orders(target_sym, params={'stop': True})
                                            except: pass
                                            
                                            time.sleep(0.5)
                                            
                                            # Cerrar posición
                                            side_close = 'sell' if side_real == 'BUY' else 'buy'
                                            params = {'reduceOnly': True}
                                            exchange.create_order(target_sym, 'market', side_close, pos_real_size, None, params)
                                            msg_binance = f"\n✅ <b>BINANCE:</b> Posición Real Cerrada."
                                        except Exception as e:
                                            msg_binance = f"\n❌ <b>BINANCE ERROR:</b> {e}"
                                    else:
                                        msg_binance = "\n👻 <b>BINANCE:</b> No había posición real."

                                    # 3. LIMPIEZA DE MEMORIA (BOT)
                                    deleted = False
                                    with memory_lock: 
                                        trades_copia = list(bot_state["active_trades"].keys())
                                        for tid in trades_copia:
                                            if bot_state["active_trades"][tid]['symbol'] == target_sym:
                                                del bot_state["active_trades"][tid]
                                                deleted = True
                                    guardar_estado()
                                    
                                    msg_final = f"🏁 <b>PROTOCOLO FINALIZADO ({target_sym})</b>" + msg_binance
                                    if deleted: msg_final += "\n🗑️ Memoria del bot limpiada."
                                    
                                    enviar_telegram(msg_final)

                                except Exception as e:
                                    enviar_telegram(f"❌ Error crítico cerrar: {e}")

			                # --- COMANDO: PING (LATENCIA) ---
                            elif texto in ['/ping', 'latencia']:
                                start = time.time()
                                try:
                                    exchange.fetch_time() 
                                    end = time.time()
                                    latency = (end - start) * 1000 # a ms
                                    
                                    icon = "🟢" if latency < 200 else "" if latency < 500 else "🔴"
                                    enviar_telegram(f"{icon} <b>PONG!</b>\nLatencia Vultr-Binance: <b>{latency:.0f} ms</b>")
                                except:
                                    enviar_telegram("❌ Error de conexión con Binance.")

     			            
                            # CHEQUEO DE BNB 
                            
                            elif texto in ['/bnb', 'gasolina', 'fees']:
                                try:
                                    # Leemos el balance de FUTUROS
                                    bal = exchange.fetch_balance()
                                    
                                    # Buscamos BNB y USDT
                                    bnb_total = bal.get('BNB', {}).get('total', 0.0)
                                    usdt_total = bal.get('USDT', {}).get('total', 0.0)
                                    
                                    # Obtenemos precio aprox del BNB para saber cuánto es en dólares
                                    try:
                                        bnb_price = exchange.fetch_ticker('BNB/USDT')['last']
                                        bnb_usd_value = bnb_total * bnb_price
                                    except:
                                        bnb_price = 0
                                        bnb_usd_value = 0
                                    
                                    msg = f"⛽ <b>ESTADO DEL TANQUE (Fees)</b>\n\n"
                                    
                                    # Semáforo
                                    if bnb_usd_value > 5:
                                        msg += f"✅ <b>NIVEL ÓPTIMO</b>\n"
                                        msg += f"Tienes: {bnb_total:.4f} BNB (~${bnb_usd_value:.2f})\n"
                                        msg += "Estás ahorrando el 10% en comisiones. 🤑"
                                    elif bnb_usd_value > 0.5:
                                        msg += f"⚠️ <b>NIVEL BAJO (RESERVA)</b>\n"
                                        msg += f"Tienes: {bnb_total:.4f} BNB (~${bnb_usd_value:.2f})\n"
                                        msg += "Te queda poca gasolina barata."
                                    else:
                                        msg += f"🔴 <b>TANQUE VACÍO (USDT MODE)</b>\n"
                                        msg += f"Tienes: {bnb_total:.4f} BNB\n"
                                        msg += "⚠️ Se están pagando comisiones en USDT (Sin descuento).\n"
                                        msg += "No afecta la operativa, solo es un poco más caro."

                                    enviar_telegram(msg)
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error leyendo BNB: {e}")

                            
                            # DIAGNÓSTICO 2 (Errores)

                            elif texto.startswith('/diagnostico'):
                                try:
                                    parts = texto.split()
                                    if len(parts) < 2:
                                        enviar_telegram("⚠️ Uso: /diagnostico STRK")
                                        continue
                                        
                                    raw_sym = parts[1].upper()
                                    sym_clean = f"{raw_sym}/USDT"
                                    
                                    msg = f"🩺 <b>DIAGNÓSTICO {raw_sym}</b>\n\n"
                                    
                                    # 1. Prueba Normal
                                    try:
                                        o1 = exchange.fetch_open_orders(sym_clean)
                                        icon = "✅" if o1 else "❌"
                                        msg += f"{icon} Normal: {len(o1)}\n"
                                    except: msg += "⚠️ Normal: Error\n"
                                    
                                    # 2. Prueba con PARÁMETRO OCULTO 
                                    try:
                                        
                                        o2 = exchange.fetch_open_orders(sym_clean, params={'stop': True})
                                        icon = "✅" if o2 else "❌"
                                        msg += f"{icon} <b>Param 'stop=True':</b> {len(o2)}\n"
                                        if o2: 
                                            msg += f"   👉 ¡Ahí están! Son {len(o2)} órdenes ocultas.\n"
                                            t = o2[0].get('type')
                                            p = o2[0].get('stopPrice')
                                            msg += f"   Ej: {t} @ {p}\n"
                                    except Exception as e: 
                                        msg += f"⚠️ Error Stop=True: {e}\n"
                                    
                                    enviar_telegram(msg)

                                except Exception as e:
                                    enviar_telegram(f"❌ Error Fatal: {e}")

                            
                            # LUPA DE ÓRDENES
                            
                            elif texto.startswith('/ordenes') or texto.startswith('/orders'):
                                try:
                                    parts = texto.split()
                                    
                                    # MODO A: ESCANEO ESPECÍFICO (Si se pasa un par)
                                    if len(parts) > 1:
                                        raw_sym = parts[1].upper()
                                        target_sym = f"{raw_sym}/USDT" if "/" not in raw_sym else raw_sym
                                        enviar_telegram(f"🔍 <b>ESCANEANDO {target_sym}...</b>")
                                        
                                        # (Normal + Stops)
                                        o_norm = exchange.fetch_open_orders(target_sym)
                                        o_stop = exchange.fetch_open_orders(target_sym, params={'stop': True})
                                        all_orders = o_norm + o_stop

                                    # MODO B: ESCANEO GLOBAL
                                    else:
                                        enviar_telegram("🔍 <b>ESCANEANDO TODO BINANCE (MODO GLOBAL)...</b>\nTrayendo todas las órdenes de una sola vez...")
                                        
                                        # 1. Traer TODAS las Limit normales
                                        try:
                                            o_norm = exchange.fetch_open_orders()
                                        except: o_norm = []
                                        
                                        # 2. Traer TODAS las Condicionales/Stops
                                        try:
                                            o_stop = exchange.fetch_open_orders(params={'stop': True})
                                        except: o_stop = []
                                        
                                        # 3. Combinar
                                        all_orders = o_norm + o_stop

                                    # --- PROCESAMIENTO Y VISUALIZACIÓN ---
                                    # ID (Por si alguna aparece en ambas listas)
                                    seen = set()
                                    unique_orders = []
                                    for o in all_orders:
                                        if o['id'] not in seen:
                                            unique_orders.append(o)
                                            seen.add(o['id'])

                                    if not unique_orders:
                                        target_txt = "el mercado" if len(parts) == 1 else target_sym
                                        enviar_telegram(f"✨ <b>LIMPIO:</b> No detecté órdenes pendientes en {target_txt}.")
                                    else:
                                        msg = f"📋 <b>ÓRDENES ENCONTRADAS ({len(unique_orders)})</b>\n\n"
                                        
                                        # Agrupar por Símbolo
                                        unique_orders.sort(key=lambda x: x['symbol'])
                                        
                                        for o in unique_orders:
                                            sym = o['symbol'].split(':')[0]
                                            oid = o['id']
                                            side = o['side'].upper() 
                                            qty = o['amount']
                                            
                                            # Detectar Tipo
                                            stop_price = o.get('stopPrice') or o.get('triggerPrice')
                                            price = o.get('price')
                                            
                                            if stop_price and float(stop_price) > 0:
                                                tipo_txt = f"Trigger: <b>{stop_price}</b>"
                                                if 'STOP' in str(o['type']).upper(): icon = "🛑 SL"
                                                elif 'TAKE' in str(o['type']).upper(): icon = "🎯 TP"
                                                else: icon = "⚠️ COND"
                                            else:
                                                tipo_txt = f"Limit: <b>{price}</b>"
                                                icon = "🔭 LMT"
                                            
                                            msg += f"🔸 <b>{sym}</b> {icon} {side}\n"
                                            msg += f"   Qty: {qty} | {tipo_txt}\n"
                                            msg += f"   ID: <code>{oid}</code>\n\n" 
                                        
                                        msg += "Para borrar: <code>/kill PAR ID</code>"
                                        if len(msg) > 4000: msg = msg[:4000] + "\n...(cortado)..."
                                        enviar_telegram(msg)

                                except Exception as e:
                                    enviar_telegram(f"❌ Error al leer órdenes: {e}")

                            
                            # CIERRE POR ID
                            
                            elif texto.startswith('/kill') or texto.startswith('/borrar'):
                                # /kill BTC 18273645
                                try:
                                    parts = texto.split()
                                    if len(parts) < 3:
                                        raise ValueError("Faltan datos.")
                                    
                                    raw_sym = parts[1].upper()
                                    order_id = parts[2]
                                    
                                    # Variantes de nombre
                                    targets = [
                                        f"{raw_sym}/USDT",          # CCXT
                                        f"{raw_sym}/USDT:USDT",     # Linear
                                        f"{raw_sym}USDT"            # Raw
                                    ]
                                    
                                    enviar_telegram(f"🔪 <b>INTENTANDO ELIMINAR {order_id}...</b>")
                                    
                                    killed = False
                                    last_err = ""
                                    
                                    # Probamos todas las combinaciones posibles
                                    for t_sym in targets:
                                        
                                        try:
                                            exchange.cancel_order(order_id, t_sym)
                                            killed = True
                                            break
                                        except: pass
                                        
                                        # STOP=TRUE
                                        try:
                                            exchange.cancel_order(order_id, t_sym, params={'stop': True})
                                            killed = True
                                            break
                                        except Exception as e:
                                            last_err = str(e)
                                    
                                    if killed:
                                        enviar_telegram("✅ <b>ORDEN ELIMINADA CON ÉXITO.</b>")
                                    else:
                                        enviar_telegram(f"❌ No se pudo eliminar.\nBinance dice: {last_err}\n(Se probó modo Normal y Stop=True en 3 variantes)")
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error sintaxis: {e}\nUso: <code>/kill STRK [ID]</code>")

                            
                            # LIMPIEZA TOTAL
                            
                            elif texto.startswith('/limpiar') or texto.startswith('/cleanorders'):
                                try:
                                    parts = texto.split()
                                    if len(parts) < 2: raise ValueError("Falta el par.")
                                    
                                    raw_sym = parts[1].upper()
                                    
                                    targets = [
                                        f"{raw_sym}/USDT",
                                        f"{raw_sym}/USDT:USDT",
                                        f"{raw_sym}USDT"
                                    ]
                                    
                                    enviar_telegram(f"🧹 <b>BARRIDO TOTAL EN {raw_sym}...</b>\nEjecutando limpieza profunda (Normal + Stops)...")
                                    
                                    hits = 0
                                    for t_sym in targets:
                                        
                                        try:
                                            exchange.cancel_all_orders(t_sym)
                                            hits += 1
                                        except: pass 
                                        
                                        # Stop=True
                                        try:
                                            exchange.cancel_all_orders(t_sym, params={'stop': True})
                                            hits += 1
                                        except: pass

                                    if hits > 0:
                                        enviar_telegram(f"✅ <b>LIMPIEZA COMPLETADA.</b>\nSe enviaron órdenes de purga a Limits y Stops.")
                                    else:
                                        enviar_telegram("⚠️ <b>ALERTA:</b> Binance no confirmó la limpieza (¿Error de conexión?).")
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error al limpiar: {e}")

                            
                            # PRECIO RÁPIDO
                            
                            elif texto.startswith('/ticker') or texto.startswith('/precio'):
                                try:
                                    sym = texto.split()[1].upper()
                                    if "/" not in sym: sym += "/USDT"
                                    ticker = exchange.fetch_ticker(sym)
                                    price = ticker['last']
                                    change = ticker['percentage']
                                    icon = "🟢" if change > 0 else "🔴"
                                    enviar_telegram(f"💲 <b>{sym}</b>: {price} ({icon} {change:.2f}%)")
                                except:
                                    enviar_telegram("⚠️ Error. Uso: /precio BTC")

			                # --- COMANDO: AJUSTAR FUSIBLE DIARIO ---
                            elif texto.startswith('/setfuse') or texto.startswith('/fusible'):
                                try:
                                    
                                    val = float(texto.split()[1])
                                    if 0.01 <= val <= 0.50:
                                        MAX_DAILY_LOSS_PCT = val
                                        limite_usd = CAPITAL_MAXIMO * MAX_DAILY_LOSS_PCT
                                        enviar_telegram(f"🔌 <b>FUSIBLE ACTUALIZADO</b>\nNuevo Límite: <b>-{int(val*100)}%</b>\n(Aprox -${limite_usd:.2f} diarios)")
                                    else:
                                        enviar_telegram("⚠️ Valor inseguro. Usa entre 0.01 (1%) y 0.50 (50%).")
                                except:
                                    enviar_telegram("⚠️ Uso: /setfuse 0.05 (para 5%)")

                            # --- COMANDO: AJUSTAR SPREAD MÁXIMO ---
                            elif texto.startswith('/setspread'):
                                try:
                                    
                                    val = float(texto.split()[1])
                                    if 0.001 <= val <= 0.05:
                                        MAX_SPREAD_ALLOWED = val
                                        enviar_telegram(f"🛡️ <b>FILTRO SPREAD ACTUALIZADO</b>\nMáximo permitido: <b>{val*100:.2f}%</b>")
                                    else:
                                        enviar_telegram("⚠️ Valor inseguro. Usa entre 0.001 (0.1%) y 0.05 (5%).")
                                except:
                                    enviar_telegram("⚠️ Uso: /setspread 0.002 (para 0.2%)")

                            
                            # BANCA INTERNA (SPOT / FUTUROS / FONDOS)
                            
                            elif texto.startswith('/bank') or texto.startswith('/transfer'):
                    
                                # Default: /bank 100  --> Mueve 100 USDT de SPOT a FUTUROS
                                
                                try:
                                    parts = texto.split()
                                    if len(parts) < 2: raise ValueError("Falta cantidad")
                                    
                                    amount = float(parts[1])
                                    
                                    # Origen y Destino por defecto
                                    origen_user = 'SPOT'
                                    destino_user = 'FUTUROS'
                                    
                                    # Si el usuario especifica
                                    if len(parts) >= 4:
                                        origen_user = parts[2].upper()
                                        destino_user = parts[3].upper()
                                    
                                    # Mapa de Traducción a Códigos Binance (CCXT)
                                    # CCXT usa: 'spot', 'future', 'funding'
                                    wallet_map = {
                                        'SPOT': 'spot',
                                        'MAIN': 'spot',
                                        
                                        'FUT': 'future',
                                        'FUTURES': 'future',
                                        'FUTUROS': 'future',
                                        'USDT-M': 'future',
                                        
                                        'FUND': 'funding',
                                        'FONDOS': 'funding',
                                        'FUNDING': 'funding',
                                        'EARN': 'funding'
                                    }
                                    
                                    # Validamos
                                    code_from = wallet_map.get(origen_user)
                                    code_to = wallet_map.get(destino_user)
                                    
                                    if not code_from or not code_to:
                                        raise ValueError(f"Billetera desconocida. Usa: SPOT, FUT, FONDOS")

                                    if code_from == code_to:
                                        raise ValueError("Origen y Destino son iguales.")

                                    enviar_telegram(f"🏦 <b>PROCESANDO TRANSFERENCIA...</b>\n{origen_user} ➡️ {destino_user}: ${amount:.2f} USDT")
                                    
                                    # EJECUTAR
                                    
                                    transfer = exchange.transfer('USDT', amount, code_from, code_to)
                                    
                                    tran_id = transfer.get('id', 'OK')
                                    enviar_telegram(f"✅ <b>TRANSFERENCIA EXITOSA</b>\nID: {tran_id}\n\nUsa /balance para verificar el saldo del Bot.")
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Transferencia: {e}\n\n<b>Uso Correcto:</b>\n<code>/bank 100 SPOT FUT</code> (Carga Bot)\n<code>/bank 50 FUT SPOT</code> (Retira Ganancia)\n<code>/bank 100 FONDOS FUT</code> (Desde P2P)")

                            
                            # BILLETERA TOTAL

                            elif texto == '/wallets' or texto == '/balance':
                                try:
                                    enviar_telegram("🏦 <b>ESCANEANDO FONDOS...</b>")
                                    
                                    # PRECIO BNB
                                    bnb_price = 0.0
                                    try:
                                        ticker_bnb = exchange.fetch_ticker('BNB/USDT')
                                        bnb_price = ticker_bnb['last']
                                    except: pass

                                    # SALDOS FUTUROS
                                    bal_fut = exchange.fetch_balance()
                                    # USDT
                                    usdt_fut_total = bal_fut['USDT']['total']
                                    usdt_fut_free = bal_fut['USDT']['free']
                                    # BNB
                                    bnb_fut_total = bal_fut.get('BNB', {}).get('total', 0.0)
                                    val_bnb_fut = bnb_fut_total * bnb_price
                                    
                                    # SALDOS SPOT
                                    spot_usdt = 0.0
                                    spot_bnb = 0.0
                                    try:
                                        exchange_spot = ccxt.binance({
                                            'apiKey': API_KEY, 'secret': API_SECRET, 
                                            'options': {'defaultType': 'spot'}
                                        })
                                        bal_spot = exchange_spot.fetch_balance()
                                        spot_usdt = bal_spot['USDT']['total']
                                        spot_bnb = bal_spot.get('BNB', {}).get('total', 0.0)
                                    except: pass
                                    val_bnb_spot = spot_bnb * bnb_price

                                    # Total Real (USDT + Valor del BNB)
                                    total_capital_usd = usdt_fut_total + spot_usdt + val_bnb_fut + val_bnb_spot

                                    msg = f"💰 <b>RESUMEN DE CUENTAS</b>\n\n"
                                    
                                    msg += f"🚀 <b>FUTUROS (Trading):</b>\n"
                                    msg += f"   💵 USDT: <b>${usdt_fut_total:.2f}</b> (Disp: ${usdt_fut_free:.2f})\n"
                                    msg += f"   ⛽ GAS BNB: <b>{bnb_fut_total:.4f} BNB</b> (~${val_bnb_fut:.2f})\n\n"
                                    
                                    msg += f"🛒 <b>SPOT (Hold):</b>\n"
                                    msg += f"   💵 USDT: <b>${spot_usdt:.2f}</b>\n"
                                    msg += f"   🪙 BNB: <b>{spot_bnb:.4f} BNB</b> (~${val_bnb_spot:.2f})\n\n"
                                    
                                    msg += f"━━━━━━━━━━━━━━━━\n"
                                    msg += f"💎 <b>CAPITAL TOTAL: ${total_capital_usd:.2f}</b>"
                                    
                                    enviar_telegram(msg)
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error Balance: {e}")

                            
                            # HOLDINGS SPOT
                            elif texto in ['/bag', '/holding', '/spotbalance']:
                                try:
                                    enviar_telegram("🎒 <b>REVISANDO MOCHILA SPOT...</b>")
                                    
                                    # Instancia Spot Temporal
                                    ex_spot = ccxt.binance({'apiKey': API_KEY, 'secret': API_SECRET, 'options': {'defaultType': 'spot'}})
                                    bal = ex_spot.fetch_balance()
                                    
                                    total_items = bal['total'] # Diccionario {'BTC': 0.01, 'ETH': 0.0 ...}
                                    found = False
                                    msg = "🎒 <b>TUS ACTIVOS EN SPOT:</b>\n\n"
                                    
                                    for coin, amount in total_items.items():
                                        if amount > 0:
                                            # Filtramos polvo
                                            if amount < 0.00001 and coin != 'BTC': continue
                                            
                                            found = True
                                            # Intentamos ver valor en USDT
                                            val_usdt = ""
                                            if coin == 'USDT':
                                                val_usdt = f"(~${amount:.2f})"
                                            else:
                                                try:
                                                    # Intento rápido de precio
                                                    price = ex_spot.fetch_ticker(f"{coin}/USDT")['last']
                                                    val_usd = price * amount
                                                    if val_usd > 1: # Solo mostramos valor si es > $1
                                                        val_usdt = f"(~${val_usd:.2f})"
                                                except: pass
                                            
                                            msg += f"🔸 <b>{coin}:</b> {amount:.6f} {val_usdt}\n"
                                    
                                    if not found:
                                        msg += "🤷‍♂️ No tienes activos (o solo polvo) en Spot."
                                    
                                    enviar_telegram(msg)
                                    
                                except Exception as e:
                                    enviar_telegram(f"❌ Error leyendo Spot: {e}")

                            
                            # VERDAD, operación en REAL, real.

                            elif texto in ['/real', 'live', 'binance']:
                                try:
                                    enviar_telegram("📡 Escaneando Binance...")
                                    positions = exchange.fetch_positions()
                                    active_pos = [p for p in positions if float(p['contracts']) > 0]
                                    
                                    if not active_pos:
                                        enviar_telegram("🧘 <b>BINANCE LIMPIO:</b> No hay posiciones abiertas.")
                                    else:
                                        msg = "🦅 <b>VISTA DE HALCÓN (DATA REAL)</b>\n"
                                        total_unrealized = 0.0
                                        
                                        # Mapa de memoria para datos
                                        memoria_map = {}
                                        with memory_lock:
                                            for k, v in bot_state["active_trades"].items():
                                                
                                                s = v['symbol'].split(':')[0].replace('/USDT', '')
                                                memoria_map[s] = v

                                        for p in active_pos:
                                            
                                            raw_sym = p['symbol']
                                            display_sym = raw_sym.split(':')[0].replace('/USDT', '')
                                            sym_key = display_sym 
                                            
                                            side = "LONG 🟢" if p['side'] == 'long' else "SHORT 🔴"
                                            size = float(p['contracts'])
                                            entry = float(p['entryPrice'])
                                            mark = float(p['markPrice']) if p['markPrice'] else entry
                                            
                                            # --- (FALLBACK A MEMORIA) ---
                                            lev = p.get('leverage')
                                            info = p.get('info', {})
                                            
                                            # 1. API Directa
                                            if lev is None: lev = info.get('leverage')
                                            if lev is None: lev = info.get('isolatedLeverage')
                                            
                                            # 2. Rescate de Memoria (Si la API falla)
                                            if lev is None and sym_key in memoria_map:
                                                lev = memoria_map[sym_key].get('leverage')
                                                
                                                if lev: lev = f"{lev}*" 
                                            
                                            if lev is None: lev = "?"
                                            
                                            # Cálculos PnL
                                            u_pnl = float(p['unrealizedPnl'])
                                            total_unrealized += u_pnl
                                            
                                            if p.get('liquidationPrice'):
                                                liq = float(p['liquidationPrice'])
                                                if mark > 0:
                                                    dist_liq = abs(mark - liq) / mark * 100
                                                    liq_txt = f"${liq:.4f} ({dist_liq:.1f}%)"
                                                else:
                                                    liq_txt = f"${liq:.4f}"
                                            else:
                                                liq_txt = "N/A"

                                            msg += f"\n🔹 <b>{display_sym}</b> {side} x{lev}\n"
                                            msg += f"   Size: {size:.3f} | PnL: ${u_pnl:.2f}\n"
                                            msg += f"   Entry: {entry} | Mark: {mark}\n"
                                            msg += f"   💀 Liq: {liq_txt}\n"
                                        
                                        icon_tot = "🤑" if total_unrealized >= 0 else "💸"
                                        msg += f"\n<b>💰 PnL FLOTANTE TOTAL: {icon_tot} ${total_unrealized:.2f}</b>"
                                        enviar_telegram(msg)

                                except Exception as e:
                                    enviar_telegram(f"❌ Error leyendo Binance: {e}")
                                    
                            
                            # HISTORIAL REAL UNIFICADO
                            elif texto.startswith('/fill') or texto.startswith('/last'):
                                try:
                                    parts = texto.split()
                                    
                                    # MODO A: Activo Específico (Rápido)
                                    # Uso: /fill BTC
                                    if len(parts) > 1:
                                        raw_sym = parts[1].upper()
                                        target_sym = f"{raw_sym}/USDT" if "/" not in raw_sym else raw_sym
                                        
                                        trades = exchange.fetch_my_trades(target_sym, limit=10)
                                        if not trades:
                                            enviar_telegram(f"📭 Sin ejecuciones recientes en {target_sym}.")
                                        else:
                                            msg = f"🧾 <b>HISTORIAL: {target_sym}</b>\n"
                                            for t in reversed(trades): # Mostramos del más reciente al más viejo
                                                side = t['side'].upper()
                                                price = t['price']
                                                qty = t['amount']
                                                
                                                # Fee display
                                                fee_txt = ""
                                                if 'fee' in t and t['fee']:
                                                    cost = t['fee'].get('cost', 0)
                                                    curr = t['fee'].get('currency', '')
                                                    if cost > 0: fee_txt = f" (Fee: -{cost:.4f} {curr})"

                                                dt = datetime.fromtimestamp(t['timestamp']/1000).strftime('%d/%m %H:%M')
                                                msg += f"• {dt} <b>{side}</b> x{qty}{fee_txt} @ {price}\n"
                                            enviar_telegram(msg)

                                    # MODO B: Escáner Global (Lento)
                                    # Uso: /fill  (sin argumentos)
                                    else:
                                        enviar_telegram("🕵️ <b>ESCANEO GLOBAL INICIADO...</b>\nRevisando historial de todas las monedas configuradas...")
                                        
                                        def scan_trades_global():
                                            all_trades = []
                                            
                                            # Función worker para buscar en paralelo
                                            def fetch_safe(s):
                                                try: return exchange.fetch_my_trades(s, limit=3) # Traemos los últimos 3 de cada una
                                                except: return []

                                            # Usamos hilos para no tardar 1 año
                                            # Escaneamos SOLO las monedas que el bot vigila (COINS_TO_TRADE)
                                            # para no golpear la API con 2000 requests.
                                            with ThreadPoolExecutor(max_workers=10) as executor:
                                                results = executor.map(fetch_safe, COINS_TO_TRADE)
                                                for r in results: all_trades.extend(r)
                                            
                                            # (El más nuevo primero)
                                            all_trades.sort(key=lambda x: x['timestamp'], reverse=True)
                                            
                                            # Top 15 Globales
                                            top_trades = all_trades[:15]
                                            
                                            if not top_trades:
                                                enviar_telegram("📭 No encontré operaciones recientes en tu lista de monedas.")
                                                return

                                            msg = "🌎 <b>ÚLTIMAS EJECUCIONES (BINANCE REAL)</b>\n"
                                            for t in top_trades:
                                                sym_clean = t['symbol'].split(':')[0]
                                                side = "🟢 BUY" if t['side'] == 'buy' else "🔴 SELL"
                                                price = t['price']
                                                qty = t['amount']
                                                dt = datetime.fromtimestamp(t['timestamp']/1000).strftime('%H:%M')
                                                
                                                msg += f"{dt} | {sym_clean} {side} | {qty} @ {price}\n"
                                            
                                            enviar_telegram(msg)

                                        # Lanzamos el escáner en un hilo aparte para no trabar al bot
                                        threading.Thread(target=scan_trades_global).start()

                                except Exception as e:
                                    enviar_telegram(f"❌ Error Fill: {e}")

                            # SALUD FINANCIERA

                            elif texto in ['/riesgo', 'risk', 'salud']:
                                try:
                                    # Obtenemos Balance Total (USDT)
                                    bal = exchange.fetch_balance()
                                    total_wallet = bal['USDT']['total']
                                    
                                    # Obtenemos el valor nocional de todo lo abierto
                                    positions = exchange.fetch_positions()
                                    notional_total = 0.0
                                    
                                    for p in positions:
                                        if float(p['contracts']) > 0:
                                            # Notional = Cuánto dinero mueve la posición realmente
                                            # Si no viene el campo 'notional', lo calculamos aprox
                                            val = float(p.get('notional', 0))
                                            if val == 0:
                                                val = float(p['contracts']) * float(p['markPrice'])
                                            notional_total += val
                                    
                                    # Cálculo de Apalancamiento Real de la Cuenta
                                    lev_real = notional_total / total_wallet if total_wallet > 0 else 0
                                    
                                    # Diagnóstico
                                    if lev_real < 3: status = "🟢 Seguro (Conservador)"
                                    elif lev_real < 10: status = "🟡 Moderado (Estándar)"
                                    elif lev_real < 20: status = "🟠 Alto (Agresivo)"
                                    else: status = "🔴 PELIGROSO (Ludópata)"
                                    
                                    msg = f"🏥 <b>SALUD DE CUENTA</b>\n\n"
                                    msg += f"💰 Balance Total: ${total_wallet:.2f}\n"
                                    msg += f"🎰 Dinero en Juego (Notional): ${notional_total:.2f}\n"
                                    msg += f"⚖️ <b>Apalancamiento Real: x{lev_real:.2f}</b>\n"
                                    msg += f"Diagnóstico: {status}"
                                    
                                    enviar_telegram(msg)
                                except Exception as e:
                                    enviar_telegram(f"❌ Error calculando riesgo: {e}")

                            # --- AYUDA (FULL COMMANDS) ---
                            elif texto in ['/help', 'ayuda']:
                                help_msg = (
                                    "👾 <b>COMANDOS STELLARIUM V6.0 (MENÚ COMPLETO)</b>\n\n"

                                    "<b>🦅 Ojos de Halcón (Binance Directo):</b>\n"
                                    "/real     - 📡 Ver posiciones REALES + Liq Price\n"
                                    "/fill [PAR]  - 🧾 Historial Real (Vacío=Scan Global / Par=Filtro)\n"
                                    "/ordenes [PAR] - 📋 Ver órdenes pendientes (Vacío=Todas / Par=Filtro)\n"
                                    "/diagnostico [PAR] - 🩺 Test API (Detectar ceguera de órdenes)\n"
                                    "/riesgo   - 🏥 Salud de cuenta y Apalancamiento\n"
                                    "/balance  - 💰 Ver saldo USDT (Total/Disp)\n"
                                    "/deposit [PAR] - 📥 Ver mi dirección [PAR] (Para depositar), USDT Default\n"
                                    "/bank [CANT] [FROM] [TO] - 🏦 Transferencia Interna\n"
                                    "<i>(Usa: SPOT, FUT, FONDOS)</i>\n"
                                    "/bnb      - ⛽ Chequear Gasolina (Fees)\n"
                                    "/wallets  - 💰 Ver USDT en Futuros, Spot y Fondos\n"
                                    "/bag      - 🎒 Ver qué monedas tienes en Spot\n"
                                    "/ping     - 📡 Latencia Real\n\n"

                                    "<b>🕹️ Operativa Manual (Soberana):</b>\n"
                                    "/operar [PAR] [SIDE] [LEV] [MARGEN] [TP] [SL]\n"
                                    "<i>Ej: /operar BTC BUY 20 100 99000 90000</i>\n"
                                    "/limit [PAR] [SIDE] [PRECIO] [LEV] [MARGEN] [TP] [SL]\n"
                                    "<i>🔭 Limit Sniper con Vigilante Auto-Protect</i>\n"
                                    "/spot [PAR] [SIDE] [USDT] - 🛒 Compra Spot (Hold)\n"
                                    "/buybnb [USDT] - ⛽ Recarga Gasolina (Spot->Fut)\n"
                                    "/edit [PAR] [SL/TP] [PRECIO] - Mover Stop/Target\n"
                                    "/kill [PAR] [ID] - 🔪 Borrar orden específica (Cirugía)\n"
                                    "/limpiar [PAR] - 🧹 Borrar TODAS las órdenes (Naked)\n"
                                    "/close [PAR]   - 👨‍🚒 Cerrar posición específica\n"
                                    "/precio [PAR]  - 💲 Ver cotización rápida\n\n"

                                    "<b>🧠 Cerebro IA & Estrategias:</b>\n"
                                    "/info [PAR] - 🔍 Análisis IA (Jefe/Soldado/HMM)\n"
                                    "/stats      - 📊 WinRate por Estrategia\n"
                                    "/sesgo      - ⚖️ Ver ajuste dinámico (Balanza)\n"
                                    "/audit      - 👮 Ver castigos y bloqueos activos\n"
                                    "/gerente    - 👔 Ver reglas del Gerente V2\n"
                                    "/mod [PAR] [STRAT] [ACCION] - Tunear Strat\n"
                                    "<i>(Acciones: VIP, UNVIP, BLOCK, RESET)</i>\n"
                                    "/resetsesgo - 🧠 Borrar memoria de aprendizaje\n\n"

                                    "<b>👁️ Visibilidad Interna (Bot):</b>\n"
                                    "/status   - 🤖 Estado General + RAM/CPU\n"
                                    "/trades   - 💼 Ver memoria de operaciones activas\n"
                                    "/pnl      - 📉 Ganancias Diarias/Totales (Real)\n"
                                    "/logs [N] - 🛠️ Ver depuración (Errores, Sizing, etc)\n"
                                    "/historial [N] - 🗄️ Ver resultados financieros (PnL)\n"
                                    "/json     - 💾 Descargar archivo de estado (.json)\n\n"

                                    "<b>⚙️ Configuración en Caliente:</b>\n"
                                    "/mode REAL/PAPER  - 🔄 Cambiar entorno\n"
                                    "/config           - ⚙️ Ver parámetros actuales\n"
                                    "/setcap [NUM]     - 💰 Capital Máximo ($)\n"
                                    "/setbase [0.XX]   - 🌱 % Base (Ej: 0.30)\n"
                                    "/setfuse [0.XX]   - 🔌 Fusible Diario (Ej: 0.05)\n"
                                    "/setspread [0.XX] - 🛡️ Max Spread (Ej: 0.002)\n"
                                    "/setbench [STRAT] [VAL] - 📏 Ajustar Techo ATR\n"
                                    "/mutesync         - 🔕 Silenciar Alertas Sync\n"
                                    "/sleep y /wake    - 💤 Pausar / ⚡ Reanudar\n\n"

                                    "<b>🚨 Zona de Peligro:</b>\n"
                                    "/cerrar   - ☢️ PÁNICO (Cierra TODO a Mercado)\n"
                                    "/sync     - 🕵️ Forzar auditoría (Limpiar Fantasmas)\n"
                                    "/reboot   - ♻️ Reiniciar contenedor\n"
                                    "/stop     - 🛑 Apagar sistema definitivamente"
                                )
                                enviar_telegram(help_msg)
                                
            time.sleep(1) 
            
	    # --- MANEJO DE ERRORES AL FINAL DE LA FUNCIÓN ---
        except requests.exceptions.ReadTimeout:
            continue # Si no hay mensajes en 20s, seguimos intentando (Normal)
            
        except requests.exceptions.ConnectionError:
            print("⚠️ Telegram: Sin conexión. Reintentando...")
            time.sleep(5)
            
        except Exception as e:
            print(f"⚠️ Error Polling Telegram: {e}")
            ultimo_update_id += 1 
            time.sleep(1)

# ==============================================================================
# DESCARGA Y FEATURES (Punto 8: Manejo Errores)
# ==============================================================================
def descargar_historial_profundo(exchange, symbol, tf=TIMEFRAME, silent=False):
    if tf == '1d': dias = DAYS_HISTORY + 50
    else: dias = DAYS_HISTORY 
    start_since = exchange.milliseconds() - (dias * 24 * 60 * 60 * 1000)
    for attempt in range(3):
        all_ohlcv = []; since = start_since; limit = 1000; failed = False
        if not silent: print(f"\n   ⬇️  {symbol} [{attempt+1}/3]: ...", end="", flush=True)
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                if not silent: print(f"\r   ⬇️  {symbol}: {len(all_ohlcv)} velas...", end="", flush=True)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < limit or since >= exchange.milliseconds() - 60000: break
                time.sleep(0.1)
            except Exception as e: 
                print(f" ❌ ERROR REAL: {e}") 
                failed = True; break
        if not failed and len(all_ohlcv) > 0:
            df = pd.DataFrame(all_ohlcv, columns=['time','open','high','low','close','volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            df.set_index('time', inplace=True)
            if not silent: print(" ✅")
            return df.sort_index()
        time.sleep(5)
    return None

def verificar_seguridad_horaria(symbol, df_5m):
    """
    Replica el filtro de entrenamiento: 
    Si Volatilidad > 20% o Cambio Brusco > 20% en 1H -> KILL SWITCH + 4 Horas Cooldown
    """
    # 1. Chequear si ya está bloqueada por un evento anterior
    if "volatility_blocklist" not in bot_state: bot_state["volatility_blocklist"] = {}
    
    block_until = bot_state["volatility_blocklist"].get(symbol)
    if block_until:
        if datetime.now() < datetime.fromisoformat(block_until):
            return False, f"⛔ Volatilidad Extrema (CoolDown hasta {block_until.split(' ')[1][:5]})"
        else:
            # Borramos el bloqueo
            del bot_state["volatility_blocklist"][symbol]
            guardar_estado()

    # 2. Resample a 1H (Igual que HMM, ignorando vela actual incompleta)
    try:
        df_1h = df_5m.resample('1h').agg({
            'open':'first', 'high':'max', 'low':'min', 'close':'last'
        }).dropna()
        
        # Eliminamos la última vela incompleta
        if len(df_1h) > 1:
            df_1h = df_1h.iloc[:-1]
            
        if len(df_1h) < 24: return True, "" # Poca data, dejamos pasar por las dudas

        # 3. Calcular Indicadores de Volatilidad
        # ATR %
        df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
        df_1h['ATR_Pct'] = df_1h['ATR'] / df_1h['close']
        
        # Log Return
        df_1h['Log_Ret'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
        
        # Miramos la última vela cerrada
        last_row = df_1h.iloc[-1]
        
        atr_pct = last_row['ATR_Pct']
        log_ret = abs(last_row['Log_Ret'])

        # 4. ¿> 20%?
        LIMIT_VOL = 0.20 # 20%
        
        if atr_pct >= LIMIT_VOL or log_ret >= LIMIT_VOL:
            # ACTIVAR BLOQUEO POR 4 HORAS
            cooldown_time = datetime.now() + timedelta(hours=4)
            bot_state["volatility_blocklist"][symbol] = str(cooldown_time)
            guardar_estado()
            
            msg = f"⛔ DETECTADO: Volatilidad {atr_pct*100:.1f}% / Cambio {log_ret*100:.1f}%. Bloqueado 4h."
            return False, msg

        return True, "" 

    except Exception as e:
        print(f"⚠️ Error verificando seguridad {symbol}: {e}")
        return True, "" 

def verificar_fusible_diario():
    """
    Calcula si la pérdida de HOY supera el % permitido del Capital Máximo.
    Return: (Status_OK, PnL_Hoy, Limite_USD)
    """
    try:
        historial = bot_state.get("closed_history", [])
        if not historial: return True, 0.0, 0.0

        today_str = datetime.now().strftime('%Y-%m-%d')
        pnl_hoy = 0.0
        
        # Sumamos solo lo de HOY y del MODO ACTUAL
        for t in historial:
            t_mode = t.get('mode', 'REAL' if not PAPER_TRADING else 'PAPER')
            current_mode = 'REAL' if not PAPER_TRADING else 'PAPER'
            
            if t_mode == current_mode and t.get('time', '').startswith(today_str):
                pnl_hoy += t.get('pnl_usd', 0.0)
        
        # CÁLCULO DINÁMICO DEL LÍMITE
        # Ej: $40 * 0.05 = $2.00. Límite es -$2.00
        limit_usd = -(CAPITAL_MAXIMO * MAX_DAILY_LOSS_PCT)
        
        # Chequeo: Si vamos perdiendo más que el límite (ej: -3.00 < -2.00)
        if pnl_hoy <= limit_usd:
            return False, pnl_hoy, limit_usd
            
        return True, pnl_hoy, limit_usd
        
    except Exception as e:
        print(f"⚠️ Error fusible: {e}")
        return True, 0.0, 0.0

def preparar_input_tactico(df_5m, scaler):
    try:
        df_1h = df_5m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        if len(df_1h) < 100: return None
        df_1h['RSI'] = ta.rsi(df_1h['close'], length=14)
        df_1h['ADX'] = ta.adx(df_1h['high'], df_1h['low'], df_1h['close'], length=14)['ADX_14']
        df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
        df_1h['ATR_Norm'] = df_1h['ATR'] / df_1h['close']
        df_1h['SMA_50'] = ta.sma(df_1h['close'], length=50)
        df_1h['Dist_SMA'] = (df_1h['close'] - df_1h['SMA_50']) / df_1h['SMA_50']
        df_1h['Slope'] = ta.slope(df_1h['close'], length=5)
        df_1h.dropna(inplace=True)
        cols = ['RSI', 'ADX', 'ATR_Norm', 'Dist_SMA', 'Slope']
        last = df_1h[cols].iloc[-48:].values
        if last.shape != (48, 5): return None
        return scaler.transform(last).reshape(1, 48, 5)
    except: return None

def preparar_input_hmm(df_5m, scaler):
    """1 Hora + Volumen + 6 Indicadores."""
    try:
        # 1. Resample 1H
        df_1h = df_5m.resample('1h').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
        if len(df_1h) > 1: df_1h = df_1h.iloc[:-1] # Ignorar vela incompleta
        if len(df_1h) < 60: return None 

        # 2. Features
        atr = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
        df_1h['ATR_Pct'] = atr / df_1h['close']
        df_1h['Log_Ret'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
        sma50 = ta.sma(df_1h['close'], length=50)
        df_1h['Dist_SMA'] = (df_1h['close'] - sma50) / sma50
        df_1h['RSI'] = ta.rsi(df_1h['close'], length=14)
        df_1h['ADX'] = ta.adx(df_1h['high'], df_1h['low'], df_1h['close'], length=14)['ADX_14']
        df_1h['Vol_Chg'] = np.log(df_1h['volume'] / (df_1h['volume'].shift(1) + 1)) 

        df_1h.dropna(inplace=True)
        if df_1h.empty: return None

        # 3. Seleccionar última
        cols_modelo = ['Log_Ret', 'ATR_Pct', 'RSI', 'ADX', 'Dist_SMA', 'Vol_Chg']
        return scaler.transform(df_1h[cols_modelo].iloc[-1:].values)

    except Exception as e:
        print(f"⚠️ Error HMM Input: {e}")
        return None

# ==============================================================================
# EJECUCIÓN Y SALIDAS (Punto 7)
# ==============================================================================

# LÓGICA DE UMBRALES DINÁMICOS (ASIMÉTRICA)

def obtener_umbrales_ajustados(lado_operacion):
    """
    Return: (Contexto, Tacticos) ajustados según racha global.
    """
    historial = bot_state.get("closed_history", [])
    
    # Valores Base
    ctx_actual = UMBRAL_CONTEXTO
    tac_actuales = UMBRALES_TACTICOS.copy()
    
    # 1. Filtro Seguridad (Mínimo 20 operaciones globales)
    ultimos_40 = historial[-40:]
    if len(ultimos_40) < 20:
        return ctx_actual, tac_actuales

    # 2. Calcular Win Rates
    longs = [t for t in ultimos_40 if t['lado'] == 'BUY'] 
    shorts = [t for t in ultimos_40 if t['lado'] == 'SELL']
    
    wr_long = sum(1 for t in longs if t['resultado'] == 'WIN') / len(longs) if longs else 0.0
    wr_short = sum(1 for t in shorts if t['resultado'] == 'WIN') / len(shorts) if shorts else 0.0
    
    delta = wr_long - wr_short 
    
    # --- ESTÁNDAR MÍNIMO DE CALIDAD ---
    # Si el WinRate es menor al 40%, no se facilita la entrada,
    # sin importar qué tan mal esté el otro lado.
    MIN_WR_FOR_BONUS = 0.40 
    
    # 3. Configuración Asimétrica
    FACTOR_RECOMPENSA = 0.12  
    FACTOR_PENALIZACION = 0.02 
    
    ajuste = 0

    if lado_operacion == 'BUY': # LONGS
        if delta > 0: # Longs son mejores que Shorts
            # Solo premiamos si Long es >40%
            if wr_long >= MIN_WR_FOR_BONUS:
                ajuste = -(delta * FACTOR_RECOMPENSA) # Bonificación
            else:
                ajuste = 0 # Neutro
        else: # Shorts son mejores
            ajuste = abs(delta) * FACTOR_PENALIZACION # Castigo
            
    elif lado_operacion == 'SELL': # SHORTS
        if delta < 0: # Shorts son mejores que Longs
            # Solo premiamos si Short >40%
            if wr_short >= MIN_WR_FOR_BONUS:
                ajuste = -(abs(delta) * FACTOR_RECOMPENSA) # Bonificación
            else:
                ajuste = 0 # Neutro
        else: # Longs son mejores
            ajuste = delta * FACTOR_PENALIZACION # Castigo
            
    # 4. Aplicar Ajuste
    
    # A) Ajuste Contexto - Impacto completo
    ctx_actual += ajuste
    
    # B) Ajuste Táctico - Impacto al 50%
    ajuste_tactico = ajuste * 0.5
    
    for k in tac_actuales:
        nuevo_valor = tac_actuales[k] + ajuste_tactico
        # --- HARD CAPS TÁCTICOS (SEGURIDAD) ---
        tac_actuales[k] = max(0.23, min(nuevo_valor, 0.40))
        
    # --- HARD CAPS CONTEXTO ---
    ctx_actual = max(0.45, min(ctx_actual, 0.60))
    
    return ctx_actual, tac_actuales

def calcular_sesgo_estrategia(strat, side):
    """
    Calcula ajuste de umbral por PnL reciente.
    INCLUYE: 
    1. "Amnistía Temporal" (Si pasó mucho tiempo sin operar).
    2. Incentivo de Recuperación (Si ganó la última, suaviza el castigo a la mitad).
    """
    history = bot_state.get("closed_history", [])
    current_mode = 'PAPER' if PAPER_TRADING else 'REAL'
    
    # 1. Filtrado de historial (Modo + Estrategia + Lado)
    relevant = [t for t in history if t.get('strat') == strat and t.get('lado') == side and t.get('mode', 'REAL') == current_mode][-SESGO_WINDOW_TRADES:]
    
    if not relevant: return 0.0
    
    # Cálculo PnL Acumulado (Suma de Dólares)
    net_pnl = sum(t.get('pnl_usd', 0.0) for t in relevant)
    ajuste = 0.0
    
    # --- LÓGICA DE BONUS / CASTIGO ---
    if net_pnl > 0:
        # RACHA GANADORA (BONUS)
        # Bajamos el umbral para facilitar entradas. Max descuente: -0.05
        ajuste = -(min(net_pnl * 0.005, 0.05))
    
    else:
        # RACHA PERDEDORA (CASTIGO)
        # Subimos el umbral para exigir más calidad. Max castigo: +0.03
        ajuste = min(abs(net_pnl) * 0.015, 0.03)
        
        # --- INCENTIVO DE RECUPERACIÓN ---
        # Si el PnL global es malo (-), pero el ÚLTIMO trade fue WIN,
        # significa que puede estar remontando. Reducimos el castigo a la mitad.
        ultimo_trade = relevant[-1]
        if ultimo_trade.get('resultado') == 'WIN':
            ajuste = ajuste / 2 
        
        # --- LÓGICA DE AMNISTÍA (Tiempo) ---
        # Solo revisamos el tiempo si hay castigo activo
        if ajuste > 0:
            last_trade_time_str = ultimo_trade.get('time', '')
            if last_trade_time_str:
                try:
                    last_date = datetime.fromisoformat(last_trade_time_str)
                    minutes_passed = (datetime.now() - last_date).total_seconds() / 60
                    
                    if minutes_passed > SESGO_PERDON_MINUTOS:
                        return 0.0 # ¡Amnistía! :)))
                except: pass

    return ajuste

def monitor_limit_order(exchange, symbol, order_id, side, lev, margin, tp, sl):
    """
    "VIGILANTE": 
    - Espera a que se llene la Limit.
    - Si falla al poner SL/TP, guarda 0 en memoria para no romper el ciclo principal.
    """
    enviar_telegram(f"🕵️ <b>VIGILANTE ACTIVO:</b> Observando orden {order_id} en {symbol}...")
    
    while True:
        try:
            # 1. Consultar estado
            order = exchange.fetch_order(order_id, symbol)
            status = order['status'] 
            
            if status == 'closed': # SE LLENÓ
                fill_price = float(order['average']) if order['average'] else float(order['price'])
                filled_qty = float(order['amount'])
                
                msg_fill = f"⚡ <b>LIMIT EJECUTADA ({symbol})</b>\nPrecio: {fill_price}\nIntentando poner SL/TP..."
                enviar_telegram(msg_fill)
                
                # Valores finales para la memoria (Request usuario por defecto)
                final_sl = sl
                final_tp = tp
                status_memoria = 'OPEN'
                
                # 2. Poner SL y TP
                try:
                    side_exit = 'sell' if side == 'BUY' else 'buy'
                    
                    # SL
                    params_sl = {'stopPrice': sl, 'reduceOnly': True}
                    exchange.create_order(symbol, 'STOP_MARKET', side_exit, filled_qty, None, params_sl)
                    
                    # TP
                    params_tp = {'stopPrice': tp, 'reduceOnly': True}
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', side_exit, filled_qty, None, params_tp)
                    
                    enviar_telegram(f"✅ <b>PROTECCIONES ACTIVAS:</b>\nSL: {sl} | TP: {tp}")
                    
                except Exception as e_cond:
                    # (Error -2021 o similar):
                    # Guardamos 0 en memoria para que el Main Loop NO cierre la operación falsamente
                    final_sl = 0.0
                    final_tp = 0.0
                    enviar_telegram(f"⚠️ <b>ERROR CRÍTICO SL/TP:</b>\n{e_cond}\n\n🛑 <b>POSICIÓN DESPROTEGIDA.</b>\nSe ha guardado sin TP/SL para evitar errores de cálculo.\n👉 <i>Usa /edit o la App para ponerlos YA.</i>")
                
                # 3. Guardar en Memoria del Bot (Safe Mode)
                trade_id = str(uuid.uuid4())[:8]
                new_trade = {
                    'symbol': symbol, 'side': side, 'amount': filled_qty,
                    'leverage': lev, 'margin_used': margin,
                    'entry_time': str(datetime.now()), 'strategy': 'LIMIT_SNIPER',
                    'price': fill_price, 
                    'sl': final_sl, # Si falló, será 0
                    'tp': final_tp, # Si falló, será 0
                    'mode': 'REAL' if not PAPER_TRADING else 'PAPER',
                    'status': status_memoria
                }
                
                with memory_lock:
                    bot_state["active_trades"][trade_id] = new_trade
                guardar_estado()
                
                break 
            
            elif status == 'canceled' or status == 'rejected':
                enviar_telegram(f"❌ <b>LIMIT CANCELADA ({symbol})</b>\nVigilante retirándose.")
                break
            
            time.sleep(15) # Espera 15s antes de volver a mirar
            
        except Exception as e:
            print(f"Error Vigilante: {e}")
            time.sleep(30)

def ejecutar_orden(exchange, symbol, estrategia, side, precio, tp, sl, meta_info="", atr_pct=None, leverage_manual=None):
    trade_id = f"{symbol}_{estrategia}"

    # Lectura segura al inicio
    with memory_lock:
        if trade_id in bot_state["active_trades"]: return
    
    inicializar_estado_estrategia(trade_id)
    lev_racha_actual = bot_state["strategy_state"][trade_id]["leverage"]
    
    lev_final = lev_racha_actual # Por defecto, respetamos la racha
    
    # --- Mensaje silencioso ---
    stellarium_msg = "" 
    
    # INTERVENCIÓN DE STELLARIUM
    if leverage_manual is not None:
        if leverage_manual == 1: # DEFENSIVO
            lev_final = 1
            print(f"🛡️ STELLARIUM INTERVIENE: Racha x{lev_racha_actual} -> Bajado a x1 por Riesgo.")
            stellarium_msg = "\n🛡️ <b>STELLARIUM:</b> 🛑 Defensa Activada (x1)"
            
        elif leverage_manual > 1: # AGRESIVO
            lev_final = max(lev_racha_actual, leverage_manual)
            if lev_final > lev_racha_actual:
                print(f"☄️ STELLARIUM POTENCIA: Base x{lev_racha_actual} -> Subido a x{lev_final} por Certeza.")
                stellarium_msg = f"\n☄️ <b>STELLARIUM:</b> 🔥 Sniper Mode (x{lev_final})"
            else:
                print(f"💎 RACHA RESPETA: Manteniendo x{lev_final} (Mayor que la sugerencia IA).")

    # 3. Guardamos el apalancamiento
    bot_state["strategy_state"][trade_id]["leverage"] = lev_final
    lev = lev_final

    # ==============================================================================
    # SIZING DOBLE FILTRO: BONUS VOLATILIDAD + LÍMITE ESTRATEGIA (Spicy)
    # ==============================================================================
    
    # ----------------------------------------------------------------------
    # 1: Techos de Estrategia
    # ----------------------------------------------------------------------
    
    strat_key = 'DEFAULT'
    for s_name in BENCHMARKS_ATR.keys():
        if s_name in estrategia: 
            strat_key = s_name
            break
            
    atr_multiplier_tp = BENCHMARKS_ATR[strat_key]

    # ----------------------------------------------------------------------
    # 2: CÁLCULO "PICANTE" (5m)
    # ----------------------------------------------------------------------
    
    capital_base = CAPITAL_MAXIMO * PORCENTAJE_BASE 
    
    # 1. REFERENCIA REALISTA
    # Usamos 0.25% como el "Stop Ideal". 
    # Si el stop real es este, usa el 100% de la base.
    sl_referencia = 0.0030
    
    distancia_sl = abs(precio - sl)
    sl_pct = distancia_sl / precio
    
    # 2. EL PISO
    # 0.001 (0.1%) para que no bloquee la referencia de 0.25%
    sl_pct_calculo = max(sl_pct, 0.001)
    
    # Cálculo lineal base
    multiplicador = sl_referencia / sl_pct_calculo
    
    # PREMIO AL RIESGO
    # Si el stop es ancho (> 0.25%), el multiplicador es < 1.0.
    # Ahí le damos el bonus para suavizar la caída de tamaño.
    if multiplicador < 1.0:
        bonus_picante = 1.25  
        multiplicador = multiplicador * bonus_picante
        # Ejemplo: Stop 0.5% -> Mult 0.5 -> Con Bonus sube a 0.625
    
    usdt_picante = capital_base * multiplicador

    # ----------------------------------------------------------------------
    # 3: EL TECHO DE LA ESTRATEGIA (Límite Máximo de Pérdida)
    # ----------------------------------------------------------------------
    # Freno del bonus si es peligroso.
    
    target_pct_estimado = atr_multiplier_tp * 0.01 
    gloria_usd = (CAPITAL_MAXIMO * 0.95) * target_pct_estimado
    
    if sl_pct > 0:
        usdt_limite_estrategia = gloria_usd / sl_pct
    else:
        usdt_limite_estrategia = 6.0
        
    # ----------------------------------------------------------------------
    # 4: LA DECISIÓN FINAL (El cruce)
    # ----------------------------------------------------------------------
    
    # El Bonus del Paso 2 intenta subir la apuesta.
    # El Límite del Paso 3 le pone un techo.
    usdt_objetivo = min(usdt_picante, usdt_limite_estrategia)
    
    # Aplicamos Techo Global y Piso Global de Binance
    techo_global = CAPITAL_MAXIMO * 0.95
    piso_minimo = 6.0
    
    usdt_final = max(piso_minimo, min(usdt_objetivo, techo_global))
    
    # Conversión a Monedas
    amount = (usdt_final * lev) / precio

    # ==============================================================================
    # PUENTE DE VARIABLES
    # ==============================================================================

    margin_usdt = usdt_final 
    factor_log = multiplicador 

    # --- Validar Mínimo ---
    try:
        # exchange.markets[symbol] ya está en memoria
        market_info = exchange.markets.get(symbol)
        
        if market_info:
            min_qty = market_info['limits']['amount']['min']
            
            # Verificación rápida de seguridad
            if amount < min_qty:
                ratio_riesgo = min_qty / amount
                if ratio_riesgo > 2.0:
                    print(f"⛔ SIZING ABORTADO: Binance pide {min_qty} (Riesgo x{ratio_riesgo:.1f}).")
                    return 
                
                print(f"⚠️ Ajuste QTY: Forzando mínimo ({min_qty})")
                amount = min_qty
                
                
                margin_usdt = (amount * precio) / lev
        else:
            # Fallback por si acaso (lento y seguro)
            market_info = exchange.market(symbol) 
            

    except Exception as e:
        print(f"⚠️ No pude leer límites de mercado: {e}")

    # Log Sizing
    print(f"⚖️ SIZING: SL {sl_pct*100:.2f}% | Factor {factor_log:.2f} | Margen ${margin_usdt:.1f}")

    # Log General
    msg_log = f"🚀 INTENTO {side} {symbol} | Strat: {estrategia} | Lev: x{lev} | Pos: ${margin_usdt:.2f}"
    print("\n" + "*"*80); print(msg_log); 
    if meta_info: print(f"    🧠 AI Context: {meta_info}")
    print("*"*80 + "\n"); 
    log_to_file(msg_log)

    msg_oportunidad = f"👀 <b>OPORTUNIDAD DETECTADA: {symbol}</b>\nTrying: ${margin_usdt:.2f}..."
    enviar_telegram(msg_oportunidad)

    if not PAPER_TRADING:
        orden_principal_ejecutada = False
        cantidad_lograda = 0

        try:
            try: exchange.set_margin_mode('ISOLATED', symbol)
            except: pass 
            
            exchange.set_leverage(lev, symbol)
            
            # 1. Precisión
            amt = exchange.amount_to_precision(symbol, amount)
            tp_final = exchange.price_to_precision(symbol, tp)
            sl_final = exchange.price_to_precision(symbol, sl)

            # 2. Disparo Principal
            order_main = exchange.create_order(symbol, 'market', side.lower(), amt)
            orden_principal_ejecutada = True
            cantidad_lograda = order_main['amount'] 

            # 3. Disparo Protecciones
            if side == 'BUY':
                exchange.create_order(symbol, 'STOP_MARKET', 'sell', cantidad_lograda, None, {'stopPrice': sl_final})
                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', cantidad_lograda, None, {'stopPrice': tp_final})
            else:
                exchange.create_order(symbol, 'STOP_MARKET', 'buy', cantidad_lograda, None, {'stopPrice': sl_final})
                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', cantidad_lograda, None, {'stopPrice': tp_final})
            
            print(f"✅ Orden y Protecciones Ejecutadas.")
            
            # --- Mensaje Telegram ---
            txt_side = "LONG ⬆️" if side == 'BUY' else "SHORT ⬇️"
            msg_exec = (
                f"✅ <b>ORDEN EJECUTADA: {symbol}</b>\n"
                f"{txt_side} | {estrategia} | x{lev}\n"
                f"📍 Entry: {precio}\n"
                f"🎯 TP: {tp_final}\n"
                f"🛡️ SL: {sl_final}"
                f"{stellarium_msg}"
            )
            enviar_telegram(msg_exec)

            # 4. Guardar Memoria

            with memory_lock:
                bot_state["active_trades"][trade_id] = {
                    'entry_time': str(datetime.now()), 
                    'price': precio, 'tp': tp, 'sl': sl, 
                    'side': side, 'symbol': symbol, 
                    'strategy': estrategia, 'leverage': lev,
                    'margin_used': margin_usdt,
                    'meta_info': meta_info,
                    'amount': cantidad_lograda 
                }
            guardar_estado()

        except Exception as e:
            print(f"❌ ERROR EN EJECUCIÓN: {e}")
            log_to_file(f"FALLO {symbol}: {e}")
            error_str = str(e)
            
            # --- ERRORES ESPECÍFICOS ---
            if "-2021" in error_str or "immediately trigger" in error_str:
                enviar_telegram(f"⚠️ <b>RECHAZO BINANCE (-2021)</b>\nVolatilidad extrema en {symbol}.")
                cooldown_time = datetime.now() + timedelta(minutes=30)
                bot_state["strategy_state"][trade_id]["cooldown_until"] = str(cooldown_time)
                guardar_estado()

            elif "-4164" in error_str or "notional" in error_str.lower():
                enviar_telegram(f"⚠️ <b>CAPITAL INSUFICIENTE (-4164)</b>\n{symbol} pide más dinero. Pausada 1h.")
                cooldown_time = datetime.now() + timedelta(minutes=60)
                bot_state["strategy_state"][trade_id]["cooldown_until"] = str(cooldown_time)
                guardar_estado()

            # --- ERROR -2019: SIN MARGEN SUFICIENTE (Cooldown 1h) ---
            elif "-2019" in error_str or "margin is insufficient" in error_str.lower():
                enviar_telegram(f"💸 <b>SALDO INSUFICIENTE (-2019)</b>\nFalta liquidez para {symbol}.\nPausada 1h para evitar spam.")
                cooldown_time = datetime.now() + timedelta(minutes=60)
                bot_state["strategy_state"][trade_id]["cooldown_until"] = str(cooldown_time)
                guardar_estado()
                print(f"❄️ Aplicando Cooldown de 60m a {trade_id} (Sin Saldo).")

            elif "Insufficient margin" not in error_str:
                 enviar_telegram(f"⚠️ <b>FALLO {symbol}</b>: {error_str}")

            # --- ROLLBACK ---
            if orden_principal_ejecutada:
                print(f"🚨 ALERTA: Posición abierta SIN protecciones. Cerrando...")
                enviar_telegram(f"🚨 <b>ROLLBACK:</b> Cerrando {symbol} por fallo en protecciones.")
                try:
                    # 1. Limpieza de cualquier orden basura que haya quedado
                    try: exchange.cancel_all_orders(symbol)
                    except: pass
                    try: exchange.cancel_all_orders(symbol, params={'stop': True})
                    except: pass
                    
                    # 2. Cerrar posición
                    side_salida = 'sell' if side == 'BUY' else 'buy'
                    exchange.create_order(symbol, 'market', side_salida, cantidad_lograda)
                    print(f"✅ Rollback exitoso.")
                except Exception as e2:
                    print(f"💀 ERROR FATAL ROLLBACK: {e2}")
                    enviar_telegram(f"💀 <b>PELIGRO:</b> Cierra {symbol} MANUALMENTE.")
            return 

    else:
        # PAPER TRADING
        txt_side = "LONG ⬆️" if side == 'BUY' else "SHORT ⬇️"
        msg_tg = (
            f"📄 <b>PAPER TRADING: {symbol}</b>\n"
            f"{txt_side} | {estrategia}\n"
            f"{stellarium_msg}"
        )
        enviar_telegram(msg_tg)

        # Escritura segura en Paper
        with memory_lock:
            bot_state["active_trades"][trade_id] = {
                'entry_time': str(datetime.now()), 'price': precio, 'tp': tp, 'sl': sl, 
                'side': side, 'symbol': symbol, 'strategy': estrategia, 'leverage': lev,
                'margin_used': margin_usdt, 'meta_info': meta_info
            }
        guardar_estado()

def verificar_salidas(exchange, df, symbol):
    """
    Monitoriza TP, SL y Mueve a Break Even.
    (Thread-Safe + Lógica de Monitorizacion)
    """
    # CANDIDATOS
    with memory_lock:
        # Lista IDs
        rel = [k for k, v in bot_state["active_trades"].items() if v.get('symbol') == symbol]
    
    if not rel: return
    
    last = df.iloc[-1]
    curr_price = last['close'] 
    
    to_rm = [] # Lista de trades para borrar al final

    for k in rel:
        # EXTRACCIÓN DE DATOS
        with memory_lock:
            # ¿El trade sigue vivo?
            if k not in bot_state["active_trades"]: continue
            # Usamos COPIA para no bloquear
            d = bot_state["active_trades"][k].copy()
        
        # --- DETECTAR SI ES REAL ---
        is_real_trade = (not PAPER_TRADING) or (d.get('mode') == 'REAL')
        
        hit_tp = (d['side']=='BUY' and last['high']>=d['tp']) or (d['side']=='SELL' and last['low']<=d['tp'])
        hit_sl = (d['side']=='BUY' and last['low']<=d['sl']) or (d['side']=='SELL' and last['high']>=d['sl'])
        
        
        # DETECTOR DE PULSO REAL 
        
        cerrado_externamente = False
        if is_real_trade:
            try:
                # Preguntamos a Binance si la posición sigue viva
                pos = exchange.fetch_positions([d['symbol']])
                size_binance = float(pos[0]['contracts']) if pos else 0.0
                if size_binance == 0:
                    cerrado_externamente = True
            except: pass 

        # --- LÓGICA DE SALIDA ---
        if hit_tp or hit_sl or cerrado_externamente:
            trade_margin = d.get('margin_used', CAPITAL_MAXIMO)
            
            # 1. Calculamos PnL
            if cerrado_externamente: precio_salida = curr_price
            else: precio_salida = d['tp'] if hit_tp else d['sl']

            pnl = (abs(precio_salida - d['price']) / d['price']) * trade_margin * d['leverage']
            
            if d['side'] == 'BUY':
                if precio_salida < d['price']: pnl = -abs(pnl)
            else:
                if precio_salida > d['price']: pnl = -abs(pnl)
            
            pnl_pct = (pnl / trade_margin) * 100
            
            # 2. Determinar Razón
            if cerrado_externamente:
                exit_reason = "MANUAL/EXTERNO"
                tipo_salida = 'WIN' if pnl > 0 else 'LOSS'
            else:
                exit_reason = "TP" if hit_tp else "STOP_LOSS"
                tipo_salida = 'LOSS' 
                umbral_fees = -(trade_margin * 0.0025) 
                if hit_tp: tipo_salida = 'WIN'
                elif hit_sl and pnl >= umbral_fees: 
                    tipo_salida = 'NEUTRAL' 
                    exit_reason = "BREAK_EVEN"
            
            # 3. Logs y Notificaciones
            try: log_trade_to_file(d['symbol'], d['strategy'], d['side'], exit_reason, pnl, pnl_pct)
            except: pass

            if is_real_trade:
                if tipo_salida == 'WIN': icono = '💰'
                elif tipo_salida == 'NEUTRAL': icono = '🛡️'
                else: icono = '💀'
                origen = " (Binance)" if cerrado_externamente else ""
                
                msg = f"{icono} <b>SALIDA{origen}: {k}</b>\nRaz: {exit_reason}\nPnL: ${pnl:.2f} ({pnl_pct:.2f}%)"
                enviar_telegram(msg)

            print(f"\n{tipo_salida}: {k} | PnL: ${pnl:.2f}\n")
            
            
            # LIMPIEZA DE ÓRDENES HUÉRFANAS
            
            if is_real_trade:
                limpieza_exitosa = False
                for intento in range(3):
                    try:
                        # 1. Intento Normal
                        exchange.cancel_all_orders(d['symbol'])
                        # 2. Intento Stop=True
                        exchange.cancel_all_orders(d['symbol'], params={'stop': True})
                        
                        limpieza_exitosa = True
                        break
                    except Exception as e:
                        time.sleep(1)
                
                if not limpieza_exitosa:
                    enviar_telegram(f"💀 <b>ALERTA CRÍTICA</b>\nNo pude cancelar el TP sobrante de {d['symbol']}.")
            
            
            # Pasamos la estrategia explícitamente para evitar errores con UUIDs
            actualizar_gestion_capital(k, tipo_salida, strategy_name=d.get('strategy'))
            
            # ACTUALIZAR HISTORIAL
            with memory_lock:
                if "closed_history" not in bot_state: bot_state["closed_history"] = []
                bot_state["closed_history"].append({
                    'lado': d['side'], 'resultado': tipo_salida, 'pnl_usd': pnl,
                    'pnl_pct': pnl_pct, 'strat': d['strategy'], 'time': str(datetime.now()),
                    'mode': 'REAL' if is_real_trade else 'PAPER'
                })
                if len(bot_state["closed_history"]) > 500:
                    bot_state["closed_history"] = bot_state["closed_history"][-500:]
            
            to_rm.append(k)
            continue

        # ==============================================================================
        # LÓGICA BREAK EVEN
        # ==============================================================================
        if not d.get('be_triggered', False):
            dist_total = abs(d['tp'] - d['price'])
            if d['side'] == 'BUY': 
                dist_recorrida = last['high'] - d['price']
                roi_actual = (curr_price - d['price']) / d['price']
            else: 
                dist_recorrida = d['price'] - last['low']
                roi_actual = (d['price'] - curr_price) / d['price']

            trigger_camino = dist_recorrida >= (dist_total * BE_TRIGGER_RATIO)
            ganancia_suficiente = roi_actual >= 0.0030

            if trigger_camino and ganancia_suficiente:
                if d['side'] == 'BUY': nuevo_sl = d['price'] * 1.0025
                else: nuevo_sl = d['price'] * 0.9975
                
                # ACTUALIZACIÓN ATÓMICA DEL SL EN MEMORIA
                with memory_lock:
                    if k in bot_state["active_trades"]:
                        bot_state["active_trades"][k]['sl'] = nuevo_sl
                        bot_state["active_trades"][k]['be_triggered'] = True
                
                guardar_estado()

                if is_real_trade:
                    threading.Thread(target=mover_sl_binance_seguro, args=(exchange, d, k, nuevo_sl)).start()
                else:
                    enviar_telegram(f"🛡️ <b>BE ACTIVADO (Simulado)</b>: {k}\nSL Virtual movido a {nuevo_sl:.4f}")

    # BORRADO FINAL
    if to_rm:
        with memory_lock:
            for k in to_rm:
                if k in bot_state["active_trades"]:
                    del bot_state["active_trades"][k]
        guardar_estado()

def mover_sl_binance_seguro(exchange, trade_data, trade_id, nuevo_sl_precio):
    """
    Mueve el SL de forma segura: Primero pone el nuevo, luego borra los viejos.
    Usa 'stop=True' para encontrar y borrar los SL viejos ocultos.
    """
    symbol = trade_data['symbol']
    cantidad = trade_data.get('amount') 
    side_cierre = 'sell' if trade_data['side'] == 'BUY' else 'buy'
    
    if not cantidad:
        enviar_telegram(f"⚠️ ERROR BE {trade_id}: Cantidad desconocida.")
        return

    try:
        # 1. Preparar precios
        sl_price = exchange.price_to_precision(symbol, nuevo_sl_precio)
        tp_price = exchange.price_to_precision(symbol, trade_data['tp'])

        # 2. COLOCAR NUEVOS PRIMERO 
        params_sl = {'stopPrice': sl_price, 'reduceOnly': True}
        nueva_orden_sl = exchange.create_order(symbol, 'STOP_MARKET', side_cierre, cantidad, None, params_sl)
        
        params_tp = {'stopPrice': tp_price, 'reduceOnly': True}
        nueva_orden_tp = exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', side_cierre, cantidad, None, params_tp)

        ids_nuevos = [nueva_orden_sl['id'], nueva_orden_tp['id']]
        
        # 3. LIMPIEZA
        # Traemos órdenes normales Y órdenes ocultas (Stop)
        ordenes_abiertas = []
        try:
            ordenes_abiertas += exchange.fetch_open_orders(symbol)
            ordenes_abiertas += exchange.fetch_open_orders(symbol, params={'stop': True})
        except Exception as e:
            print(f"⚠️ Error leyendo órdenes para borrar: {e}")

        # ID
        seen_ids = set()
        unique_orders = []
        for o in ordenes_abiertas:
            if o['id'] not in seen_ids:
                unique_orders.append(o)
                seen_ids.add(o['id'])
        
        borradas = 0
        for orden in unique_orders:
            if orden['id'] not in ids_nuevos:
                deleted = False
                # Intento A: Normal
                try:
                    exchange.cancel_order(orden['id'], symbol)
                    deleted = True
                except: pass
                
                # Intento B: Con Stop=True (Si falló el A)
                if not deleted:
                    try:
                        exchange.cancel_order(orden['id'], symbol, params={'stop': True})
                        deleted = True
                    except: pass
                
                if deleted: borradas += 1

        # --- ÉXITO ---
        enviar_telegram(f"🛡️ AUTO-BE ÉXITO: {trade_id}\nSL movido a {sl_price} (Limpiadas: {borradas})")
        print(f"✅ AUTO-BE Exitoso para {trade_id}")
        
    except Exception as e:
        print(f"❌ FALLO AUTO-BE: {e}")
        enviar_telegram(f"🚨 FALLO AUTO-BE: {trade_id}\n{str(e)[:40]}")

# ==============================================================================
# CONTROL MANUAL Y PÁNICO
# ==============================================================================
bot_running = True

def cerrar_todas_posiciones(exchange):
    """
    "PANIC BUTTON":
    - Cierra todas las posiciones de la IA.
    - Respeta las operaciones MANUALES y LIMITS (No las toca ni las olvida).
    - Usa el tamaño real de Binance para asegurar el cierre.
    """
    print("\n" + "🚨"*20)
    print("¡ALERTA! INICIANDO PROTOCOLO DE CIERRE MASIVO (SOLO IA)...")
    print("🚨"*20 + "\n")
    
    # 1. FOTO de la memoria
    with memory_lock:
        trades_activos = list(bot_state["active_trades"].keys())
    
    if not trades_activos:
        print("✅ No hay posiciones en memoria para cerrar.")
        return

    # 2. ESCANEO de lo real
    real_positions = {}
    if not PAPER_TRADING:
        try:
            raw_pos = exchange.fetch_positions()
            # Mapeamos Symbol -> Datos (Size, Side, etc)
            for p in raw_pos:
                if float(p['contracts']) > 0:
                    # Limpiamos nombre: 'BTC/USDT:USDT' -> 'BTC/USDT'
                    sym_clean = p['symbol'].split(':')[0]
                    real_positions[sym_clean] = p
        except Exception as e:
            print(f"⚠️ Error leyendo posiciones reales: {e}")

    for trade_id in trades_activos:
        # Lectura
        with memory_lock:
            if trade_id not in bot_state["active_trades"]: continue
            trade_data = bot_state["active_trades"][trade_id].copy()

        symbol = trade_data['symbol']
        strat = trade_data.get('strategy', 'UNKNOWN')
        
        # --- FILTRO DE PROTECCIÓN MANUAL ---
        # Si es Manual o Limit, se mantiene y en memoria.
        if 'MANUAL' in strat or 'LIMIT' in strat:
            print(f"🛡️ SALTANDO {symbol} ({strat}) - Es Soberana/Manual.")
            continue
        
        # --- LÓGICA DE CIERRE IA ---
        print(f"🧨 Cerrando IA: {symbol} ({strat})...")
        
        borrar_de_memoria = False
        
        if not PAPER_TRADING:
            # Buscamos la posición real para este par
            pos_real = real_positions.get(symbol)
            
            if pos_real:
                # EXISTE EN BINANCE -> CERRARLA CON DATOS REALES
                try:
                    # 1. Borrar órdenes pendientes (Normales y Stops)
                    try: exchange.cancel_all_orders(symbol)
                    except: pass
                    try: exchange.cancel_all_orders(symbol, params={'stop': True})
                    except: pass
                    
                    # 2. Cerrar posición exacta
                    size_real = float(pos_real['contracts'])
                    side_real = 'BUY' if pos_real['side'] == 'long' else 'SELL'
                    side_cierre = 'sell' if side_real == 'BUY' else 'buy'
                    
                    params = {'reduceOnly': True}
                    exchange.create_order(symbol, 'market', side_cierre, size_real, None, params)
                    print(f"✅ {symbol} CERRADO (Size Real: {size_real}).")
                    borrar_de_memoria = True
                    
                except Exception as e:
                    print(f"❌ ERROR CERRANDO {symbol}: {e}")
                    # Si falla, NO borramos de memoria para que el Sync intente arreglarlo luego
            else:
                # NO EXISTE EN BINANCE (Fantasma)
                print(f"👻 {symbol} ya no estaba en Binance. Limpiando memoria.")
                borrar_de_memoria = True
        else:
            # MODO PAPER
            print(f"✅ {symbol} CERRADO (Simulado).")
            borrar_de_memoria = True
        
        # 3. Borrado seguro (Solo si se cerró o no existía)
        if borrar_de_memoria:
            with memory_lock:
                if trade_id in bot_state["active_trades"]:
                    del bot_state["active_trades"][trade_id]
    
    guardar_estado()
    print("\n🏁 PROTOCOLO FINALIZADO.\n")
    enviar_telegram("🚨 <b>CIERRE MASIVO EJECUTADO</b>\nSe cerraron las posiciones de IA.\nLas manuales se han respetado.")

def escuchar_teclado(exchange):
    """Hilo que corre en paralelo escuchando comandos."""
    global bot_running
    print("\n⌨️  SISTEMA DE COMANDOS ACTIVADO:")
    print("    Escribe 'salir'  -> Para detener el bot al finalizar el ciclo.")
    print("    Escribe 'cerrar' -> PANIC BUTTON (Cierra todo y sigue corriendo).")
    print("    Escribe 'exit'   -> PANIC BUTTON + APAGAR.\n")
    
    while bot_running:
        try:
            
            cmd = input() 
            if cmd.strip().lower() == 'salir':
                print("\n🛑 SOLICITUD DE DETENCIÓN RECIBIDA. Terminando ciclo actual...")
                bot_running = False
                break
            elif cmd.strip().lower() == 'cerrar':
                cerrar_todas_posiciones(exchange)
            elif cmd.strip().lower() == 'exit':
                cerrar_todas_posiciones(exchange)
                print("\n🛑 APAGANDO SISTEMA...")
                bot_running = False
                break
        except EOFError:
            break

def sincronizar_cartera_real(exchange):
    """
    AUDITORÍA:
    - 1-3: (Aliens, Fantasmas, Desnudas).
    - 4: 'Órdenes Basura' (Huérfanas) en pares sin posición.
    """
    reporte = []
    
    try:
        # --- 0: PREPARACIÓN ---
        mapa_conocidos = {}
        tiempos_entrada = {} 

        with memory_lock:
            snapshot_trades = bot_state["active_trades"].copy()
        
        for k, v in snapshot_trades.items():
            raw = v['symbol'].split(':')[0].replace('/', '')
            mapa_conocidos[raw] = k 
            try: tiempos_entrada[raw] = datetime.fromisoformat(v['entry_time'])
            except: tiempos_entrada[raw] = datetime.now()

        # --- 1: TRAER POSICIONES REALES ---
        raw_positions = exchange.fetch_positions()
        simbolos_reales_activos = set() 

        # --- 2: AUDITORÍA DE LO QUE EXISTE EN BINANCE ---
        for p in raw_positions:
            if float(p['contracts']) > 0:
                symbol_raw = p['symbol'] 
                symbol_clean = symbol_raw.split(':')[0]
                sym_key = symbol_clean.replace('/', '') 
                
                simbolos_reales_activos.add(sym_key)
                side_pos = 'BUY' if p['side'] == 'long' else 'SELL'
                
                if sym_key in tiempos_entrada:
                    delta = (datetime.now() - tiempos_entrada[sym_key]).total_seconds()
                    if delta < 60: continue 

                # A) ALIENs
                if sym_key not in mapa_conocidos:
                    if not PAPER_TRADING:
                        reporte.append(f"👽 <b>ALIEN DETECTADO:</b> {display_sym(symbol_raw)} {side_pos} x{p['leverage']}")
                    else:
                        reporte.append(f"⚠️ <b>ALERTA REAL (ALIEN):</b> {display_sym(symbol_raw)} existe en Binance.")
                else:
                    # B) CONFLICTOS
                    trade_id = mapa_conocidos[sym_key]
                    datos_bot = bot_state["active_trades"][trade_id]
                    if datos_bot['side'] != side_pos:
                        reporte.append(f"⚠️ <b>CONFLICTO:</b> {display_sym(symbol_raw)} Bot:{datos_bot['side']} vs Bin:{side_pos}")

                # C) DESNUDAS
                ordenes_encontradas = []
                try:
                    # Normales
                    ordenes_encontradas += exchange.fetch_open_orders(symbol_clean)
                    # Ocultas
                    ordenes_encontradas += exchange.fetch_open_orders(symbol_clean, params={'stop': True})
                except: pass

                # Deduplicamos por ID
                ordenes_unicas = {o['id']: o for o in ordenes_encontradas}.values()
                
                lado_proteccion = 'sell' if side_pos == 'BUY' else 'buy'
                
                def es_proteccion(o, tipo_proteccion):
                    if o['side'].lower() != lado_proteccion: return False
                    otype = str(o.get('type', '')).upper()
                    
                    if tipo_proteccion == 'STOP':
                        if 'STOP' in otype: return True
                        trigger = float(o.get('stopPrice', 0) or o.get('triggerPrice', 0) or 0)
                        if trigger > 0: return True
                        
                    if tipo_proteccion == 'TAKE_PROFIT':
                        if 'TAKE_PROFIT' in otype: return True
                        if 'LIMIT' in otype: return True
                    return False

                has_sl = any(es_proteccion(o, 'STOP') for o in ordenes_unicas)
                has_tp = any(es_proteccion(o, 'TAKE_PROFIT') for o in ordenes_unicas)
                
                if not has_sl and not has_tp: 
                    reporte.append(f"💀 <b>PELIGRO:</b> {display_sym(symbol_raw)} DESNUDA.")
                elif not has_sl: 
                    reporte.append(f"⚠️ <b>ALERTA:</b> {display_sym(symbol_raw)} SIN SL.")

        # --- 3: INVESTIGACIÓN DE FANTASMAS ---
        if not PAPER_TRADING:
            active_ids = list(snapshot_trades.keys())
            
            for trade_id in active_ids:
                with memory_lock:
                    if trade_id not in bot_state["active_trades"]: continue
                    data = bot_state["active_trades"][trade_id].copy()

                sym_bot_clean = data['symbol'].split(':')[0].replace('/', '')
                symbol_real = data['symbol']
                
                # PROTECCIÓN LIMIT + STOPS
                tiene_ordenes_pendientes = False
                try:
                    # stop=True
                    ords = exchange.fetch_open_orders(data['symbol'], params={'stop': True})
                    if not ords: # Por si es Limit normal
                        ords = exchange.fetch_open_orders(data['symbol'])
                    if len(ords) > 0: tiene_ordenes_pendientes = True
                except: pass

                if (sym_bot_clean not in simbolos_reales_activos) and (not tiene_ordenes_pendientes):
                    
                    try: exchange.cancel_all_orders(symbol_real)
                    except: pass
                    try:
                        ticker = exchange.fetch_ticker(symbol_real)
                        curr_price = ticker['last']
                        entry = data['price']; lev = data['leverage']; margin = data.get('margin_used', 0)
                        side_mult = 1 if data['side'] == 'BUY' else -1
                        pnl_pct = ((curr_price - entry) / entry) * side_mult * lev * 100
                        pnl_usd = (margin * pnl_pct) / 100
                        
                        emoji = "💰" if pnl_pct > 0 else "❌"
                        res_tag = "WIN" if pnl_pct > 0 else "LOSS"
                            
                        msg = (f"{emoji} <b>CIERRE EXTERNO (SYNC): {display_sym(symbol_real)}</b>\n"
                               f"Posición cerrada. Limpieza 🧹.\n"
                               f"PnL Est: {pnl_pct:.2f}% (${pnl_usd:.2f})")
                        reporte.append(msg)
                        
                        if "closed_history" not in bot_state: bot_state["closed_history"] = []
                        bot_state["closed_history"].append({
                            'lado': data['side'], 'resultado': res_tag, 'pnl_usd': pnl_usd,
                            'pnl_pct': pnl_pct, 'strat': data['strategy'], 'time': str(datetime.now()),
                            'mode': 'REAL', 'nota': 'Ghost'
                        })
                    except:
                        reporte.append(f"👻 <b>FANTASMA LIMPIADO:</b> {trade_id}")

                    with memory_lock:
                        if trade_id in bot_state["active_trades"]:
                            del bot_state["active_trades"][trade_id]

        # --- 4: DETECCIÓN DE BASURA (Huérfanas) ---
        # Órdenes en símbolos donde no hay posición ni trade activo.
        if not PAPER_TRADING:
            try:
                # Traer todo
                all_limits = exchange.fetch_open_orders()
                all_stops = exchange.fetch_open_orders(params={'stop': True})
                total_orders = all_limits + all_stops
                
                basura_detectada = {} # {Symbol: Cantidad}
                
                for o in total_orders:
                    # 'ZEC/USDT:USDT' -> 'ZEC'
                    sym_clean = o['symbol'].split(':')[0].replace('/', '')
                    
                    # SIN POSICIÓN REAL Y SIN TRADE EN BOT...
                    if (sym_clean not in simbolos_reales_activos) and (sym_clean not in mapa_conocidos):
                        basura_detectada[sym_clean] = basura_detectada.get(sym_clean, 0) + 1
                
                # Reporte
                for s, count in basura_detectada.items():
                    reporte.append(f"🗑️ <b>BASURA DETECTADA:</b> {s} tiene {count} órdenes huérfanas.\n👉 Usa <code>/limpiar {s}</code>")
                    
            except Exception as e:
                print(f"⚠️ Error scan basura: {e}")

        guardar_estado()
        return "\n\n".join(reporte) if reporte else "✅ <b>SISTEMA SINCRONIZADO.</b>"
        
    except Exception as e:
        return f"❌ Error Auditoría: {e}"

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main_loop():
    # --- INICIALIZACIÓN ---
    exchange = inicializar_exchange()
    cargar_estado()
    if not PAPER_TRADING: print("\n⚠️ MODO REAL ACTIVADO ⚠️"); time.sleep(3)

    print("🧠 Cargando Modelos...")
    modelos = {}; scalers = {}
    try:
        # Carga de modelos
        for k in CONFIG_STRAT.keys(): modelos[k] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, f'model_IA_{k}.keras'))
        modelos['CONTEXTO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_CONTEXTO.keras'))
        modelos['TACTICO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_TACTICO_1H.keras'))
        modelos['HMM'] = joblib.load(os.path.join(DIR_MODELOS, 'model_hmm_unsupervised.pkl'))
        if os.path.exists(os.path.join(DIR_MODELOS, 'model_IA_DIARIO.keras')): modelos['DIARIO'] = tf.keras.models.load_model(os.path.join(DIR_MODELOS, 'model_IA_DIARIO.keras'))
        scalers['micro'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_micro_global.pkl'))
        scalers['macro'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_macro_global.pkl'))
        if os.path.exists(os.path.join(DIR_SCALERS, 'scaler_tactico_1h.pkl')): scalers['tactico'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_tactico_1h.pkl'))
        scalers['hmm'] = joblib.load(os.path.join(DIR_SCALERS, 'scaler_hmm.pkl'))
    except Exception as e: print(f"❌ Error carga: {e}"); return

    # --- BLOQUE IA STELLARIUM ---
    print("☄️ATOMIZANDO IA STELLARIUM...")

    global gerente, strat_map_inv, umbrales_pro, bot_running 

    gerente = None
    strat_map_inv = {} 
    umbrales_pro = {}  

    try:
        # 1. Cargar Modelo y Mapas
        gerente = joblib.load(os.path.join(DIR_MODELOS, 'meta_model_manager_v2.pkl'))
        strat_map_raw = joblib.load(os.path.join(DIR_MODELOS, 'meta_model_strat_map.pkl'))
        
        strat_map_inv = {v: k for k, v in strat_map_raw.items()}
        
        # 2. Cargar Umbrales Asimétricos
        with open(os.path.join(DIR_MODELOS, 'config_umbrales_pro.json'), 'r') as f:
            umbrales_pro = json.load(f)
            
        print("✅ Gerente V2 en su puesto. Configuración PRO cargada.")
    except FileNotFoundError as fnf:
        print(f"⚠️ AVISO: No se encontró al Gerente ({fnf}). El bot operará sin filtro V3.")
        gerente = None
    except Exception as e:
        print(f"❌ Error al cargar el Gerente: {e}")
        gerente = None

    data_cache = {}
    print("📥 Sincronizando datos...")
    for sym in set(TOP20_SYMBOLS + COINS_TO_TRADE):
        df = descargar_historial_profundo(exchange, sym, '5m')
        if df is not None: data_cache[sym] = df

    # Carga de Mercados

    print("📥 Cargando información de mercados (Precisiones y Límites)...")
    try:
        
        exchange.load_markets()
    except Exception as e:
        print(f"⚠️ Error cargando mercados: {e}")


    print(f"\n🤖 BOT ONLINE | Modo: {'PAPER' if PAPER_TRADING else 'REAL'}")
    log_to_file(f"=== BOT INICIADO: {datetime.now()} ===")

    # --- HILO DE COMANDOS ---
    t = threading.Thread(target=escuchar_teclado, args=(exchange,))
    t.daemon = True 
    t.start()

    # Telegram 
    t_tg = threading.Thread(target=escuchar_telegram, args=(exchange,))
    t_tg.daemon = True; t_tg.start()

    msg_inicio = f"🤖 <b>BOT V3.0 INICIADO</b>\nModo: {'PAPER' if PAPER_TRADING else 'REAL'}\nLímites: {MAX_TRADES_GLOBAL} Global / {MAX_TRADES_STRAT} Strat"
    enviar_telegram(msg_inicio)
    # ==========================================================================
    # BUCLE PRINCIPAL
    # ==========================================================================
    # --- AUDITORÍA INICIAL ---
    # Esto revisa si hay posiciones reales abiertas al reiniciar.
    if not PAPER_TRADING:
        print("🕵️ Realizando auditoría inicial de cartera...")
        try:
            # Resultado
            print(sincronizar_cartera_real(exchange))
        except: pass
    # Contador para el while
    ciclos = 0

    while bot_running:  
        print(f"\n⏱️  {datetime.now().strftime('%H:%M:%S')} | Activos: {len(bot_state['active_trades'])}")
        # --- UMBRALES DINÁMICOS ---
        
        ctx_long, tac_long = obtener_umbrales_ajustados('BUY')
        ctx_short, tac_short = obtener_umbrales_ajustados('SELL')
        
        
        # print(f" 📊 Dinámico | LONG Req: {ctx_long:.2f} | SHORT Req: {ctx_short:.2f}")  # Debug

        
        # --- DASHBOARD Y GESTIÓN DE SALIDAS ---
        
        # 1: Copia segura de la memoria (foto)
        with memory_lock:
            trades_snapshot = bot_state["active_trades"].copy()

        # 2: Usamos la foto para imprimir
        if trades_snapshot:
            print(f"🔰 --- CARTERA ACTIVA ---")
            for tid, tdata in trades_snapshot.items(): # <--- COPIA
                curr_price = tdata['price'] 
                
                # (lectura rápida)
                if tdata['symbol'] in data_cache:
                    curr_price = data_cache[tdata['symbol']]['close'].iloc[-1]
                
                side_mult = 1 if tdata['side'] == 'BUY' else -1
                pnl_pct = ((curr_price - tdata['price']) / tdata['price']) * side_mult * tdata['leverage'] * 100
                
                margin_info = f"{tdata.get('margin_used', 0):.1f}$"
                print(f"   👉 {tdata['symbol']} | {tdata['side']} | PnL: {pnl_pct:.2f}% | Strat: {tdata['strategy']} | {margin_info}")
                print(f"      🧠 {tdata.get('meta_info', 'No Info')}")
            print("-" * 60)
        
        # --- 2. DESCARGA DE DATOS ---
        try:
            mkt_idx, btc_series = None, None
            symbols_to_sync = list(data_cache.keys())

            def worker_download(sym):
                try:
                    
                    time.sleep(np.random.uniform(0.05, 0.2)) 
                    new = exchange.fetch_ohlcv(sym, TIMEFRAME, limit=30)
                    if new:
                        df_n = pd.DataFrame(new, columns=['time','open','high','low','close','volume'])
                        df_n['time'] = pd.to_datetime(df_n['time'], unit='ms', utc=True)
                        df_n.set_index('time', inplace=True); df_n.dropna(inplace=True)
                        
                        # thread-safe
                        if sym not in data_cache:
                            data_cache[sym] = df_n
                        else:
                            data_cache[sym] = pd.concat([data_cache[sym], df_n])
                            data_cache[sym] = data_cache[sym][~data_cache[sym].index.duplicated(keep='last')]

                        if len(data_cache[sym]) > 86000:
                            data_cache[sym] = data_cache[sym].iloc[-86000:]
                except Exception: 
                    pass 

            # (Max 5 workers)
            # Esto evita el error 418/429 de Binance
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(worker_download, symbols_to_sync)

            # 3. VERIFICACIÓN DE SALIDAS
            # Se ejecuta después de descargar
            for sym in symbols_to_sync:
                if sym in COINS_TO_TRADE and sym in data_cache:
                    verificar_salidas(exchange, data_cache[sym], sym)
            

            # AUTO-SYNC

            ciclos += 1
            if ciclos >= 3 and not PAPER_TRADING:
                print("\n🕵️ Ejecutando Auto-Auditoría periódica...")
                try:
                    reporte = sincronizar_cartera_real(exchange)
                    
                    # Alerta, solo si hay problemas Y no está silenciado
                    if "FANTASMA" in reporte or "ALIEN" in reporte or "PELIGRO" in reporte or "CONFLICTO" in reporte:
                        print(f"🚨 Auto-Sync encontró problemas: {reporte}")
                        
                        if not SILENCIAR_AUTO_SYNC: 
                            enviar_telegram(f"🚨 <b>AUTO-SYNC DETECTÓ CAMBIOS:</b>\n{reporte}")
                    else:
                        print("✅ Auditoría limpia.")
                        
                except Exception as e:
                    print(f"⚠️ Fallo en Auto-Sync: {e}")
                
                ciclos = 0 
            

            # Calculo Mercado
            try: 
                dr = {s.replace('/',''): data_cache[s]['close'].pct_change() for s in TOP20_SYMBOLS if s in data_cache}
                dv = {s.replace('/',''): data_cache[s]['volume'] for s in TOP20_SYMBOLS if s in data_cache}
                if dr:
                    mkt_idx = (pd.DataFrame(dr).ffill().fillna(0) * pd.DataFrame(dv).ffill().fillna(0)).sum(axis=1) / (pd.DataFrame(dv).ffill().fillna(0).sum(axis=1) + 1e-9)
                    if 'BTC/USDT' in data_cache: btc_series = data_cache['BTC/USDT']['close']
            except: pass

            if mkt_idx is None: print("   ⏳ Esperando datos..."); time.sleep(10); continue

            # --- (SLEEP) ---
            if bot_paused:
                print(f"💤 BOT EN PAUSA (SLEEP) - Gestionando salidas, ignorando entradas.")
                
                for sym in list(data_cache.keys()):
                     if sym in COINS_TO_TRADE: verificar_salidas(exchange, data_cache[sym], sym)
                
                # Esperamos un minuto y volvemos al inicio del bucle
                
                for _ in range(60):
                    if not bot_running: break
                    time.sleep(1)
                continue 
            

            # --- RECOLECCIÓN DE CANDIDATOS ---
	        
            sistema_ok, pnl_hoy, limit_usd = verificar_fusible_diario()
            
            if not sistema_ok:
                print(f"🛑 FUSIBLE DETONADO: Pérdida ${pnl_hoy:.2f} supera límite de ${limit_usd:.2f} ({int(MAX_DAILY_LOSS_PCT*100)}%).")
                print("   💤 Durmiendo 5 minutos...")
                time.sleep(300)
                continue 
            
            candidatos = [] 

            for symbol in COINS_TO_TRADE:
                if symbol not in data_cache: continue
                str_mac, str_tac, str_hmm, str_stat = "...", "...", "...", ""
                
                # --- ANÁLISIS DE SEGURIDAD (VOLATILIDAD 1H) ---
                # Usamos el dataframe de 5m
                es_seguro, motivo_seguridad = verificar_seguridad_horaria(symbol, data_cache[symbol])
                
                if not es_seguro:
                    
                    print(f"{symbol:<10} | {motivo_seguridad}"); continue

	        # FILTRO DE SPREAD
                try:
                    # Puntas de precio
                    if symbol in data_cache:
                        
                        tick = exchange.fetch_ticker(symbol)
                        ask = tick['ask']
                        bid = tick['bid']
                        
                        if ask and bid:
                            spread_pct = (ask - bid) / ask
                            if spread_pct > MAX_SPREAD_ALLOWED:
                                # Solo imprime si es muy alto
                                print(f"{symbol:<10} | ⛔ Spread Alto ({spread_pct*100:.2f}% > {MAX_SPREAD_ALLOWED*100:.2f}%)")
                                continue 
                except Exception as e_spread:
                    pass 

                df = data_cache[symbol].copy()
                # --- VARIABLES INICIALES (Para que no falle si hay error) ---
                btc_trend_score_val = 0.0
                reg_tac, conf_tac = 0, 0.0 
                reg_mac, conf_mac = 0, 0.0
                clust, str_hmm = 2, "C2"

                try:
                    df = market_context.agregar_indicadores_contexto(df, mkt_idx, btc_series)
                    for mod in [frpv, mean_reversion, breakout, simple_trend]: df = mod.aplicar_estrategia(df)
                    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                    df['ATR_Pct'] = df['ATR'] / df['close']
                    df['RSI'] = ta.rsi(df['close'], length=14)
                    
                    curr_idx = -2; row = df.iloc[curr_idx]
                    
                    # 1. Inferencia Contexto (Macro)
                    w_mac = df.iloc[curr_idx-13:curr_idx+1][COLS_MACRO].values
                    w_mac_s = scalers['macro'].transform(w_mac).reshape(1,14,len(COLS_MACRO))
                    p_mac = modelos['CONTEXTO'].predict(w_mac_s, verbose=0)[0]
                    reg_mac = np.argmax(p_mac); conf_mac = p_mac[reg_mac]
                    str_mac = f"{['RAN','BULL','BEAR','CAOS'][reg_mac]} ({conf_mac:.2f})"

                    # 2. Inferencia Táctico (NN 1H)
                    X_tac = preparar_input_tactico(df, scalers['tactico'])
                    if X_tac is not None:
                        p_tac = modelos['TACTICO'].predict(X_tac, verbose=0)[0]
                        reg_tac = np.argmax(p_tac); conf_tac = p_tac[reg_tac]
                        str_tac = f"{['RAN','BULL','BEAR','CAOS'][reg_tac]} ({conf_tac:.2f})"
                    else: str_tac = "Falta Data"

                    # 3. Inferencia HMM (Cluster)
                    X_hmm = preparar_input_hmm(df, scalers['hmm'])
                    if X_hmm is not None:
                        clust = modelos['HMM'].predict(X_hmm)[0]
                        str_hmm = f"C{clust}"
                        bot_state["market_cluster"] = int(clust)
                    
                    # Captura de datos crudos finales para STELLARIUM:
                    btc_trend_score_val = float(row.get('BTC_Trend_Score', 0.0))
                    atr_pct_val = float(row.get('ATR_Pct', 0.0))
                    rsi_val = float(row.get('RSI', 50.0))

                except: str_mac="Err"; str_tac="Err"; str_hmm="Err"; continue
                
                
                # En Telegram mostramos lo que piensan las IAs
                if "ai_views" not in bot_state: bot_state["ai_views"] = {}
                
                bot_state["ai_views"][symbol] = {
                    "MACRO": str_mac,  
                    "TACTICO": str_tac,
                    "UPDATE": datetime.now().strftime('%H:%M')
                }
                

                # --- DETERMINAR ESTRATEGIAS (UMBRALES DINÁMICOS) ---
                allow_std = []; side_allow_std = 'BOTH'
                macro_bloqueado = False
                
                # 1. SELECCIÓN DE VARA DE MEDIR (MACRO)
                # Si el Macro dice BULL (1), usamos la vara ajustada de Longs.
                # Si dice BEAR (2), la de Shorts. Si es Rango/Caos, la Estándar.
                if reg_mac == 1: ref_umbral_ctx = ctx_long
                elif reg_mac == 2: ref_umbral_ctx = ctx_short
                else: ref_umbral_ctx = UMBRAL_CONTEXTO

                # 2. BLOQUEO MACRO
                if conf_mac < ref_umbral_ctx: macro_bloqueado = True; str_stat="⛔ Duda Mac"
                elif reg_mac == 3: macro_bloqueado = True; str_stat="⛔ Caos Mac"
                
                # 3. REGLAS ESTÁNDAR (Tácticos Dinámicos)
                if not macro_bloqueado and reg_tac is not None:
                    
                    # Selección de vara de medir (TÁCTICO)
                    # Si Táctico es Bull (1) -> Usamos ajuste Long. Bear (2) -> Ajuste Short.
                    if reg_tac == 1: ref_dic_tac = tac_long
                    elif reg_tac == 2: ref_dic_tac = tac_short
                    else: ref_dic_tac = UMBRALES_TACTICOS # 0 y 3 usan el base

                    
                    if conf_tac >= ref_dic_tac.get(reg_tac, 0.5):
                        
                        if reg_tac == 0: # Rango
                            allow_std = ['RANGO']
                            if reg_mac != 0: allow_std.append('FRPV') 
                        elif reg_tac == 1: # Bull
                            allow_std = ['FRPV', 'TREND', 'BREAKOUT', 'RANGO']
                            side_allow_std = 'BUY'
                            if reg_mac == 2: allow_std = [] 
                        elif reg_tac == 2: # Bear
                            allow_std = ['FRPV', 'TREND', 'BREAKOUT', 'RANGO']
                            side_allow_std = 'SELL'
                            if reg_mac == 1: allow_std = [] 
                        elif reg_tac == 3: # Caos
                            allow_std = ['BREAKOUT', 'FRPV']
                            if reg_mac == 1: side_allow_std = 'BUY'
                            elif reg_mac == 2: side_allow_std = 'SELL'
                            elif reg_mac == 0: allow_std = ['FRPV']

                # Reglas VIP HMM
                allow_vip = []
                try:
                    X_hmm = preparar_input_hmm(df, scalers['hmm'])
                    if X_hmm is not None:
                        clust = modelos['HMM'].predict(X_hmm)[0]
                        str_hmm = f"C{clust}"
                        bot_state["market_cluster"] = int(clust)
                        if clust in REGLAS_HMM and REGLAS_HMM[clust]:
                            allow_vip = REGLAS_HMM[clust]
                except: str_hmm="Err"

                # Unificamos reglas crudas
                raw_rules = list(set(allow_std + allow_vip))
                
                # Si no hay reglas
                if not raw_rules:
                    if not "⛔" in str_stat: str_stat = "💤 Esperando"
                    print(f"{symbol:<10} | {str_mac:<15} | {str_tac:<15} | {str_hmm:<10} | {str_stat}"); continue
                
                str_stat = "" 

                # --- EVALUACIÓN (LONG/SHORT) ---
                # Iteramos sobre las estrategias base
                for base_strat in CONFIG_STRAT.keys():
                    
                    # 1. INTERPRETACIÓN DE PERMISOS
                    # Permiso Total: Si la regla es 'FRPV' -> Habilita ambos lados
                    permiso_total = base_strat in raw_rules
                    # Permiso Específico: Si la regla es 'FRPV_LONG' -> Solo Long
                    permiso_long = permiso_total or (f"{base_strat}_LONG" in raw_rules)
                    permiso_short = permiso_total or (f"{base_strat}_SHORT" in raw_rules)
                    
                    # Si no tiene permiso ni de long ni de short
                    if not (permiso_long or permiso_short): continue

                    # 2. Check GLOBAL Cooldown (Punto 7)
                    if base_strat in bot_state.get("global_strat_perf", {}):
                        cool_until = bot_state["global_strat_perf"][base_strat]["cooldown_until"]
                        if cool_until and datetime.now() < datetime.fromisoformat(cool_until):
                            continue # Estrategia castigada globalmente

                    # 3. Límite por Estrategia (Punto 4)
                    count_strat = sum(1 for k in bot_state["active_trades"] if base_strat in k)
                    if count_strat >= MAX_TRADES_STRAT: continue

                    # 4. Check Cooldown Individual
                    key_strat = f"{symbol}_{base_strat}"
                    if check_cooldown(key_strat): continue
                    if key_strat in bot_state["active_trades"]: continue

                    # 5. OBTENCIÓN DE SEÑALES CRUDAS Y FILTRADO DIRECCIONAL
                    cfg = CONFIG_STRAT[base_strat]
                    
                    
                    # Solo leemos la señal de compra si tenemos permiso_long = True
                    is_buy = (not pd.isna(row.get(cfg['buy']))) and permiso_long
                    # Solo leemos la señal de venta si tenemos permiso_short = True
                    is_sell = (not pd.isna(row.get(cfg['sell']))) and permiso_short
                    
                    if not (is_buy or is_sell): continue
                    
                    
                    # Ajustamos nombre variable
                    strat = base_strat 
                    
                    # FILTROS MACRO
                    # Si la estrategia entró por una regla VIP
                    es_vip_long = f"{base_strat}_LONG" in allow_vip or base_strat in allow_vip
                    es_vip_short = f"{base_strat}_SHORT" in allow_vip or base_strat in allow_vip
                    
                    if is_buy and not es_vip_long:
                        if macro_bloqueado: continue
                        if side_allow_std == 'SELL': continue
                        if reg_mac == 2: continue 
                    
                    if is_sell and not es_vip_short:
                        if macro_bloqueado: continue
                        if side_allow_std == 'BUY': continue
                        if reg_mac == 1: continue 

                    side = 'BUY' if is_buy else 'SELL'; direct = 1.0 if is_buy else -1.0

                    # Predicción IA
                    try:
                        w_mic = df.iloc[curr_idx-59:curr_idx+1][COLS_MICRO].values
                        w_mic_s = scalers['micro'].transform(w_mic).reshape(1,60,6)
                        d_t = np.full((1,60,1), direct)
                        X_mic = np.concatenate([w_mic_s, d_t], axis=2)

                        prob = modelos[strat].predict([X_mic, w_mac_s], verbose=0)[0][0]
                    except: continue
                    
                    # ----------------------------------------------------------
                    # UMBRAL FINAL (Dinámico + VIP + Sesgo Strat)
                    # ----------------------------------------------------------
                    umbral_real = cfg['umbral']
                    
                    # Ajuste por Sesgo de Estrategia (PnL Reciente)
                    sesgo_strat = calcular_sesgo_estrategia(strat, side)
                    umbral_real += sesgo_strat
                    
                    # Ajuste VIP (Racha o HMM)
                    es_vip_active = (is_buy and es_vip_long) or (is_sell and es_vip_short)
                    vip_until_date = bot_state["strategy_state"][key_strat].get("vip_until")
                    es_vip_racha = vip_until_date and datetime.now() < datetime.fromisoformat(vip_until_date)
                    
                    if es_vip_active or es_vip_racha: 
                        umbral_real -= VIP_UMBRAL_DISCOUNT
                    
                    
                    prefix = "🔥" if (es_vip_active or es_vip_racha) else "⚡"
                    if sesgo_strat < 0: prefix += "🟢" # Icono si tiene bonus por ganar
                    elif sesgo_strat > 0: prefix += "🛡️" # Icono si tiene castigo por perder
                    
                    pasa_umbral = prob >= umbral_real
                    estado_icon = "✅" if pasa_umbral else "❌"
                    
                    # Mostramos el umbral real exigido en el log
                    str_stat += f"{prefix}{strat[:2]} {side}({prob:.2f}/{umbral_real:.2f}){estado_icon} "
                    
                    if pasa_umbral:
                        entry = float(row['close']); atr = float(row['ATR'])
                        mtp = cfg['tp']
                        if side=='BUY': tp=entry+(atr*mtp); sl=entry-(atr*cfg['sl'])
                        else: tp=entry-(atr*mtp); sl=entry+(atr*cfg['sl'])
                        
                        if abs(entry-sl)/entry > 0.05: # Filtro volatilidad extrema
                            str_stat = str_stat.replace("✅", "⚠️Vol"); continue

                        # --- GUARDAMOS CANDIDATO ---
                        str_stat += "🚀 "
                        meta = f"M:{str_mac.split('(')[0]} T:{str_tac.split('(')[0]} H:{str_hmm} P:{prob:.2f} VIP:{es_vip_active}"
                        
                        candidatos.append({
                            'prob': prob,
                            'symbol': symbol,
                            'strat': strat,
                            'side': side,
                            'entry': entry, 'tp': tp, 'sl': sl,
                            'meta': meta,
                            # --- PARA IA STELLARIUM ---
                            'prob_ia': prob,
                            'raw_hmm': int(clust),
                            'raw_ctx': reg_mac,
                            'raw_ctx_prob': conf_mac,
                            'raw_tac': reg_tac,
                            'raw_tac_prob': conf_tac,
                            'atr_pct': atr_pct_val,
                            'rsi': rsi_val,
                            'btc_trend': btc_trend_score_val,
                            'hour': datetime.now().hour,
                            'day': datetime.now().weekday() 
                            # -----------------------------------------------
                        })
                
                if not candidatos and "✅" not in str_stat: 
                      print(f"{symbol:<10} | {str_mac:<15} | {str_tac:<15} | {str_hmm:<10} | {str_stat}")

            # --- EJECUCIÓN FINAL ---
            candidatos.sort(key=lambda x: x['prob'], reverse=True)
            
            seen_symbols = set()
            final_ops = []
            

            total_open = len(bot_state["active_trades"])
            espacio_disponible = MAX_TRADES_GLOBAL - total_open
            
            if candidatos:
                print(f"\n🔎 Analizando {len(candidatos)} candidatos con Gerente V2...")

            for c in candidatos:
                if c['symbol'] in seen_symbols or espacio_disponible <= 0: continue

                # --- EVITAR CONFLICTO EN MISMO PAR ---
                # Verificación de si ya existe alguna posición abierta en par, sin importar la estrategia
                existe_posicion_par = False
                direccion_existente = None
                
                for k, v in bot_state["active_trades"].items():
                    if v['symbol'] == c['symbol']:
                        existe_posicion_par = True
                        direccion_existente = v['side']
                        break
                
                # Si ya hay posición en el par:
                if existe_posicion_par:
                    # A: Bloquear
                    # continue 
                    
                    # B: Permitir solo si es la misma dirección
                    if direccion_existente != c['side']:
                        print(f"⚠️ {c['symbol']} descartado: Conflicto de dirección (Ya hay {direccion_existente})")
                        continue

                # --- LIMITES ESTRATEGIA ---
                active_strat = sum(1 for k in bot_state["active_trades"] if c['strat'] in k)
                pending_strat = sum(1 for op in final_ops if op['strat'] == c['strat'])
                
                if (active_strat + pending_strat) >= MAX_TRADES_STRAT: continue

                # --- LIMITES DIRECCIONALES ---
                # Cuántas hay abiertas de este lado (BUY o SELL)
                active_side = sum(1 for v in bot_state["active_trades"].values() if v['side'] == c['side'])
                
                # Cuántas vamos a abrir ahora de este lado
                pending_side = sum(1 for op in final_ops if op['side'] == c['side'])
                
                # Si sumamos más de 6, se descarta
                if (active_side + pending_side) >= MAX_TRADES_SIDE:
                    continue
                
                # Por defecto: No sugerimos nada(racha)
                lev_sugerido = None 

                # A) Si el STELLARIUM existe:
                if gerente is not None:
                    try:
                        # 1. IDs
                        strat_id = strat_map_inv.get(c['strat'], -1)
                        side_id = 1 if c['side'] == 'BUY' else 0
                        
                        # 2. Vector de Features
                        features_gerente = np.array([[
                            strat_id, side_id,
                            c['prob_ia'], c['raw_hmm'], c['raw_ctx'], c['raw_ctx_prob'], 
                            c['raw_tac'], c['raw_tac_prob'], c['atr_pct'], c['rsi'], 
                            c['btc_trend'], c['hour'], c['day']
                        ]])
                        
                        # 3. Predicción
                        prob_exito = gerente.predict_proba(features_gerente)[:, 1][0]
                        
                        # 4. Umbrales
                        cfg_strat = umbrales_pro.get(c['strat'], {})
                        cfg_side = cfg_strat.get(c['side'], {"veto": 0.30, "agresivo": 0.99})
                        u_veto = cfg_side['veto']
                        u_agg = cfg_side['agresivo']

                        # 5. SEMÁFORO DE APALANCAMIENTO
                        
                        if prob_exito < u_veto:
                            # ZONA ROJA: DEFENSIVO (Forzamos x1)
                            lev_sugerido = 1
                            print(f"🛡️ GERENTE: {c['symbol']} {c['strat']} Riesgoso ({prob_exito:.2f}). Sugiere x1.")
                            c['meta'] += f" [DEFENSA x1]"
                        
                        elif prob_exito >= u_agg:
                            # ZONA VERDE: SNIPER (Sugerimos x12)
                            lev_sugerido = 12
                            print(f"🔥 GERENTE: {c['symbol']} {c['strat']} ALTA CERTEZA ({prob_exito:.2f}). Sugiere x12!")
                            c['meta'] += f" [SNIPER x12]"
                            
                        else:
                            # ZONA AMARILLA: NORMAL (No tocamos nada)
                            lev_sugerido = None 
                            print(f"⚖️ GERENTE: {c['symbol']} {c['strat']} Normal ({prob_exito:.2f}). Respeta Racha.")

                    except Exception as e:
                        print(f"⚠️ Error Gerente {c['symbol']}: {e}. Operando normal.")
                        lev_sugerido = None

                # Guardamos la sugerencia para ejecutar_orden
                c['leverage_manual'] = lev_sugerido
                
                final_ops.append(c)
                seen_symbols.add(c['symbol'])
                espacio_disponible -= 1
            
            # --- EJECUCIÓN ---
            if final_ops:
                print(f"🚀 Ejecutando {len(final_ops)} operaciones...")
                for op in final_ops:
                     ejecutar_orden(
                         exchange, op['symbol'], op['strat'], op['side'], 
                         op['entry'], op['tp'], op['sl'], op['meta'], 
                         atr_pct=op.get('atr_pct'),
                         leverage_manual=op['leverage_manual'] # <--- SUGERENCIA
                     )
                
                if len(final_ops) > espacio_disponible:
                    print(f"⚠️ Límite Global ({MAX_TRADES_GLOBAL}) alcanzado. Se descartaron {len(final_ops) - espacio_disponible} operaciones.")

        except Exception as e: 
            error_msg = f"⚠️ <b>ERROR EN CICLO MAIN</b>\n{str(e)}"
            print(f"\n❌ Error Main: {e}"); traceback.print_exc()
            
            # Avisar a Telegram (el bot sigue corriendo)
            enviar_telegram(error_msg)
        # --- ESPERA ---
        # Si hay operaciones abiertas, chequeamos más rápido (cada 10s) para gestionar salidas
        # Si no hay nada, chequeamos cada 60s para ahorrar API y CPU.
        tiempo_espera = 20 if bot_state["active_trades"] else 60
        
        print(f"   ⏳ Ciclo terminado. Esperando {tiempo_espera}s...")
        
        # Bucle de espera
        for _ in range(tiempo_espera):
            if not bot_running: break
            time.sleep(1)
            
    print("👋 BOT DETENIDO CORRECTAMENTE.")

if __name__ == "__main__":
    main_loop()