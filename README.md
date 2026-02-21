# 🌌 Stellarium AI: Advanced Quantitative Trading System

![Python 3.10](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![Status](https://img.shields.io/badge/Status-Production-green?style=for-the-badge)

**Stellarium AI** is an autonomous, high-frequency algorithmic trading engine designed for cryptocurrency markets. It implements a **Hybrid AI Architecture** that combines Unsupervised Learning (HMM) for market regime detection, Deep Learning for pattern recognition, and Gradient Boosting/Ensemble methods for trade quality assurance.

> **Key Differentiator:** Unlike standard bots, Stellarium features a "Manager" meta-model that vets every trade signal against historical probabilities, dynamically adjusting leverage based on confidence levels and current market regimes.

## 🏗️ System Architecture

The system follows a modular pipeline within a containerized environment:

1. **Market Data Stream:** Async data ingestion from Binance.
2. **Feature Engineering:** Calculation of technical indicators and market context.
3. **Multi-Model Consensus:** - Unsupervised AI (HMM) detects the current market regime.
   - Ensemble Models generate tactical trade signals.
4. **The Manager (Meta-Model):** A secondary AI layer that evaluates the probability of a signal's success, acting as a strict risk filter.
5. **Execution Engine:** Dynamic position sizing and order execution via API.

## ⚙️ The AI Training Pipeline (Step-by-Step)

To train the models from scratch on your local machine, follow this exact execution order:

### Phase 1: Data Acquisition
* `python descargar_data.py`: Downloads the Top 20 cryptocurrencies (Market Reference/Indicator).
* `python descargar_data_universal.py`: Downloads the main asset dataset. You can easily modify the asset list within this file while keeping the strict structure.

### Phase 2: Feature Engineering & Context
* The system uses `MODULOS/labeling_objetivo.py` and `MODULOS/market_context.py` to calculate technical indicators, establish the market context, and create the target labels for the AI.

### Phase 3: Data Preparation
* `python training/prepare_multitarget_data.py`: Cleans, normalizes, and packages the generated data into `.npz` arrays ready for neural network ingestion.

### Phase 4: Model Training & Meta-Model Generation
* `python training/analisis_unsupervised_hmm_v2.py`: Trains the Hidden Markov Model to recognize latent market states (Bull, Bear, Ranging, High Volatility).
* `python training/train_ensemble.py`: Trains the core predictive models (The Analysts) on the prepared data.
* `python training/minero_datos_masivo.py`: **(The Simulator)** Runs the trained ensemble over 4 years of historical data to generate a massive dataset (`DATASET_GERENTE_MASIVO_V3.csv`) detailing every AI success and failure.
* `python training/entrenar_gerente.py`: Trains the Meta-Model (The Manager) using the simulated dataset to learn when to trust or veto the ensemble's signals.

## 📂 Repository Structure

```text
├── main.py                     # Production Orchestrator
├── descargar_data.py           # Top 20 Market Data Downloader
├── descargar_data_universal.py # Universal/Custom Asset Downloader
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container Configuration
│
├── MODULOS/                    # Core Logic & Feature Engineering
│   ├── labeling_objetivo.py
│   └── market_context.py
│
├── ESTRATEGIAS/                # Technical Strategies & Indicators
│
├── training/                   # AI Lab (Data Mining & Training)
│   ├── analisis_unsupervised_hmm_v2.py
│   ├── prepare_multitarget_data.py
│   ├── train_ensemble.py
│   ├── entrenar_gerente.py
│   ├── calibrar_agresividad_fina.py
│   └── minero_datos_masivo.py
│
├── backtesting/                # Validation Engines
│   ├── backtest_engine.py
│   └── audit_predictive_power.py
│
└── IA_FINAL_CHECKS/            # System Integrity & Calibration
    ├── check_hmm.py
    └── hyper_calibration_matrix.py
```

## 🚀 Setup & Installation
Prerequisites: Docker & Docker Compose (Recommended) or Python 3.10+.

1. Clone the Repo

```Bash
git clone https://github.com/xdutk/trading-bot-ia-engine.git
cd trading-bot-ia-engine
```
2. Environment Configuration
Create a .env file in the root directory (this is explicitly ignored by Git for security):

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

3. Run with Docker

```Bash
docker build -t stellarium-bot .
docker run -d --env-file .env --name stellarium-v1 stellarium-bot
```

## 🕹️ Usage & Execution
Once your models are trained, you can run the system in either simulation or live mode. Ensure your .env file is properly configured with Binance and Telegram credentials before proceeding.

### 1. Backtesting (Validation)
Before risking real capital, validate your AI models against historical data:

```bash
python backtesting/backtest_engine.py
```
This engine simulates the bot's precise behavior over historical data. It outputs a detailed console report including Win Rate, Profit Factor, and PnL by Strategy, and automatically generates an equity curve chart (resultado_backtest_v14.png) to visualize performance.

### 2. Live Trading (Production)
To start the trading engine, run the main orchestrator:

```Bash
python main.py
```

* Operational Modes:

- Paper Trading (Default): The bot processes live market data and logs simulated trades. Controlled by the PAPER_TRADING = True flag in main.py.

- Real Money: The bot executes real orders via the Binance API. Set PAPER_TRADING = False or switch it dynamically via Telegram.

### 📱 Telegram Command & Control (C2)

Stellarium AI features a robust remote control system via Telegram. Once main.py is running, send /help to your bot to access the full suite of commands. Key features include:

- /status: Check system uptime, RAM/CPU usage, and active trades.

- /mode REAL / /mode PAPER: Hot-swap between simulation and live trading.

- /pnl: View real-time financial results.

- /cerrar: Panic Button. Instantly closes all AI-managed positions at market price.

## ⚠️ Disclaimer & Usage
Models: Pre-trained models (.keras, .pkl, etc.) and historical data .csv files are intentionally excluded from this repository to protect IP and reduce size. Use the scripts in the training/ pipeline to generate your own models locally.

Educational Use: This software is for educational and portfolio demonstration purposes. Trading cryptocurrency futures involves significant financial risk.

Author: Xavier Dutka
Python Developer | AI & Quantitative Trading Enthusiast
