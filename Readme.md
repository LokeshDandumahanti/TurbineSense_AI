# ⚙️ TurbineSense AI  
### GenAI-Powered Turbine Health Monitoring, Emission Forecasting & Predictive Maintenance

<img width="1918" height="975" alt="Screenshot 2025-11-01 232327" src="https://github.com/user-attachments/assets/e228cc14-8d78-436a-a120-730fbcd3581f" />

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

<img width="1522" height="617" alt="Screenshot 2025-11-01 232623" src="https://github.com/user-attachments/assets/9356c0b8-2f48-4aed-a0f9-b402ff6c4b0d" />

**TurbineSense AI enables:**
- 🔍 Continuous TEY (Thermal Efficiency Yield) monitoring  
- 🌫️ CO & NOx emission forecasting  
- ⚙️ Predictive maintenance via drift/anomaly alerting  
- 🤖 AI-assisted troubleshooting using LLM knowledge  

---<img width="1371" height="595" alt="Screenshot 2025-11-01 232501" src="https://github.com/user-attachments/assets/93343004-c4a3-49c0-90c5-ec51c9416e84" />
<img width="1377" height="703" alt="Screenshot 2025-11-01 232436" src="https://github.com/user-attachments/assets/20e219dc-8250-439c-b52f-27ed9db3125e" />
<img width="1372" height="600" alt="Screenshot 2025-11-01 232417" src="https://github.com/user-attachments/assets/53e0fd5f-b7ec-4058-8c30-862d686cf042" />
<img width="1375" height="776" alt="Screenshot 2025-11-01 232354" src="https://github.com/user-attachments/assets/00f59674-8916-4e7c-a8fd-d679ef99dbf5" />
<img width="1522" height="617" alt="Screenshot 2025-11-01 232623" src="https://github.com/user-attachments/assets/c92067e0-9f55-4723-9d59-7a97bef14e0a" />
<img width="1375" height="555" alt="Screenshot 2025-11-01 232525" src="https://github.com/user-attachments/assets/656bc41a-0207-46c3-9c48-26d96f418ca5" />
<img width="1370" height="681" alt="Screenshot 2025-11-01 232514" src="https://github.com/user-attachments/assets/bd7decf6-f6c6-4bb0-bdc8-cc97fbe4b912" />


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




