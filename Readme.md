# ⚙️ TurbineSense AI  
### GenAI-Powered Turbine Health Monitoring, Emission Forecasting & Predictive Maintenance

TurbineSense AI is an end-to-end AI system for **real-time gas turbine performance monitoring**, **emission prediction**, **anomaly detection**, and **operator decision support using Generative AI**.  
It integrates **machine learning, time-adaptive models, and an LLM-based troubleshoot assistant** into a unified Streamlit dashboard.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Capabilities](#key-capabilities)  
3. [System Architecture](#system-architecture)  
4. [Prediction & Drift Logic](#prediction--drift-logic)  
5. [Tech Stack](#tech-stack)  
6. [How to Run](#how-to-run)  
7. [AI Troubleshoot Chatbot](#ai-troubleshoot-chatbot)  
8. [Visual Outputs](#visual-outputs)  
9. [Future Enhancements](#future-enhancements)  
10. [License](#license)

---

## 🚀 Project Overview
Gas turbines operate under extreme thermal and mechanical stress, making **performance degradation and emission drift** inevitable. Traditional monitoring methods struggle to provide early warnings before efficiency drops or regulatory limits are violated.

**TurbineSense AI enables:**
- 🔍 Continuous TEY (Thermal Efficiency Yield) monitoring  
- 🌫️ CO & NOx emission forecasting  
- ⚙️ Predictive maintenance via drift/anomaly alerting  
- 🤖 AI-assisted troubleshooting using LLM knowledge  

---

## ✅ Key Capabilities

| Feature | Description |
|---------|-------------|
| 🔧 Predictive Modeling | TEY, CO, NOx forecast using ML |
| ⏳ Short-term Model | Learns recent operating behavior (rolling window) |
| 📈 Long-term Model | Models multi-year historical turbine patterns |
| ⚠️ Drift & Anomaly Alerts | Deviation tracking against baseline & trend |
| 🤖 LLM Chatbot | Operator assistant powered by Groq LLM |
| 📊 Interactive Dashboard | Streamlit UI with live plots & logs |
| 🔁 Continuous Learning | Progressive retraining on new batches |
| 🛠️ Single-File Deployment | Lightweight architecture (`app.py`) |

---

## 🏗 System Architecture

┌──────────────────────────────┐
│ Streamlit UI │ ← Dashboard + Chatbot
└────────────┬─────────────────┘
│
┌────────────▼──────────────┐
│ Prediction Engine │
│ • Short-term XGBoost │
│ • Long-term XGBoost │
└────────────┬──────────────┘
│
┌────────────▼──────────────┐
│ Drift & Anomaly Logic │
│ • Residual Tracking │
│ • Threshold Evaluation │
└────────────┬──────────────┘
│
┌────────────▼──────────────┐
│ LLM Troubleshoot Agent │ ← Optional Groq API
└───────────────────────────┘

---

## 🔬 Prediction & Drift Logic

### Rolling Training Cycle
1. Train **short-term model** on latest 30 rows  
2. Predict next 30 rows using:  
   - ✅ Long-term model (baseline behavior)  
   - ✅ Short-term model (recent behavior)  
3. Compare predictions vs actual  
4. Repeat → mimics daily turbine monitoring

### Interpretation Logic

| Short-Term | Long-Term | Meaning |
|------------|-----------|---------|
| ✅ Stable | ✅ Stable | Turbine healthy |
| ⚠️ Deviates | ✅ Stable | Local fluctuation |
| ⚠️ Deviates | ⚠️ Deviates | System drift / degradation |
| 🚨 Strong deviation | 🚨 Strong deviation | Critical anomaly – immediate action |

---

## 🧩 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| ML Models | XGBoost (Short & Long term) |
| AI Assistant | Groq LLM API |
| Data Layer | Pandas, NumPy |
| Visualization | matplotlib |
| Deployment | Local / Cloud (single script) |

---

## 💻 How to Run

### ✅ 1. Install Dependencies
```bash
pip install -r requirements.txt
python app.py
streamlit app.py



