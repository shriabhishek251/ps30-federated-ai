# 🛡️ FedShield-AI: Privacy-Preserving Clinical AI Platform

A decentralized, privacy-preserving machine learning platform designed for clinical diagnostic networks.

FedShield-AI enables multiple healthcare institutions to collaboratively train a shared predictive model **without exchanging raw Electronic Health Records (EHRs)**.

The platform simulates a realistic hospital federation where each node contains heterogeneous (**non-IID**) patient populations. It integrates:

- **Federated Averaging (FedAvg)**
- **Differential Privacy using DP-SGD**
- **Cumulative Rényi Differential Privacy Accounting**
- **Zero-Trust Pairwise Masked Secure Aggregation**
- **Fault-Tolerant Federated Orchestration**
- **Interactive Streamlit Governance Dashboard**

The goal is to demonstrate how clinical AI systems can achieve collaborative intelligence while maintaining privacy, security, and regulatory compliance.

---

# 📊 Benchmark Summary

All models are evaluated centrally on the **same held-out test split**:

- Dataset: CDC Diabetes Health Indicators
- Test samples: **50,736**
- Split: 20% held-out partition
- Decision threshold: **0.50**

| Configuration | Accuracy | F1 Score | Precision | Recall | ROC-AUC | Description |
|---|---:|---:|---:|---:|---:|---|
| **Centralized Baseline** | 0.7282 | 0.4436 | 0.3103 | 0.7778 | **0.8262** | Centralized training with complete data access |
| **Federated (DP-Off)** | 0.8179 | 0.4593 | 0.3917 | 0.5550 | **0.8244** | 4 Non-IID clients, 10 FedAvg rounds |
| **Fed + DP (ε = 8.0)** | 0.8650 | 0.2260 | 0.5600 | 0.1410 | **0.8200** | Relaxed privacy setting |
| **Fed + DP (ε = 4.0)** | 0.8649 | 0.1988 | 0.5732 | 0.1202 | **0.8178** | Balanced privacy regime |
| **Fed + DP (ε = 1.0)** | 0.8650 | 0.1940 | 0.5760 | 0.1170 | **0.8160** | Strict privacy guarantee |

---

# 🚀 Key Architectural Features

## 1. Distributed Federated Learning

Implemented using Flower's modern:

- `ServerApp`
- `ClientApp`

architecture.

The system simulates **4 hospital clients** where each node trains locally and only exchanges model updates.

### Non-IID Data Simulation

Clinical distributions are simulated using:

- Dirichlet partitioning
- α = 0.5

This creates realistic hospital differences such as:

- varying disease prevalence
- demographic imbalance
- heterogeneous patient cohorts

---

# 2. Differential Privacy Engine

FedShield-AI integrates **Opacus DP-SGD**.

Privacy protection is achieved through:

### Per-sample Gradient Clipping

```
max_grad_norm = 1.0
```

Every individual patient's gradient contribution is bounded before aggregation.

### Gaussian Noise Injection

Calibrated Gaussian noise is added to gradients to prevent reconstruction attacks.

---

# 3. Cumulative Rényi Differential Privacy Accounting

Instead of tracking privacy cost independently per round, FedShield-AI performs:

- multi-round composition
- epoch-level accounting
- complete training lifespan privacy estimation

This provides mathematically rigorous privacy guarantees.

---

# 4. Zero-Trust Secure Aggregation

Implemented in:

```
src/secure_agg.py
```

Clients apply pairwise masks before transmitting updates.

Individual client updates appear as random noise to the server.

The masking mechanism:

\[
y_i = x_i + \sum_{j>i} r_{ij} - \sum_{j<i} r_{ji}
\]

When aggregated:

\[
\sum_i y_i = \sum_i x_i
\]

The paired masks cancel algebraically, allowing exact model aggregation without exposing individual hospital updates.

---

# 5. Fault-Tolerant Federated Training

Hospital networks may experience failures.

FedShield-AI simulates catastrophic client failure:

Example:

- Client 1 disconnects during Round 5

The server automatically:

- detects node failure
- removes unavailable client
- recalculates aggregation weights
- continues training without interruption

---

# 6. Streamlit Governance Dashboard

The interactive dashboard provides:

- privacy budget visualization
- training loss curves
- benchmark comparisons
- secure aggregation verification logs
- model performance analytics

Dashboard entry point:

```
dashboard/app.py
```

---

# 🏗️ Project Structure

```
ps30-federated-ai/

├── pyproject.toml
├── requirements.txt
├── .gitignore

├── data/
│   └── Raw datasets and generated client partitions

├── metrics/
│   ├── baseline.json
│   ├── dp_sweep_summary.json
│   └── federated_history_*.json


├── src/

│   ├── data_loader.py
│   │   └── Non-IID partitioning and leakage-free splitting

│   ├── model.py
│   │   └── Shared PyTorch MLP architecture

│   ├── baseline.py
│   │   └── Centralized training pipeline

│   ├── threshold_analysis.py
│   │   └── Precision/Recall threshold calibration

│   └── secure_agg.py
│       └── Pairwise masking implementation


├── fl_app/

│   ├── task.py
│   │   └── DP-SGD, training tasks, evaluation

│   ├── client_app.py
│   │   └── Federated client logic

│   └── server_app.py
│       └── FedAvg orchestration


└── dashboard/

    └── app.py
        └── Streamlit governance dashboard
```

---

# 💻 Installation & Quickstart

## 1. Environment Setup

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPO_URL>

cd ps30-federated-ai
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 2. Data Preparation & Baseline Verification

Download dataset, create scaler, and generate client shards:

```bash
python src/data_loader.py
```

Run centralized baseline:

```bash
python src/baseline.py
```

Expected:

```
ROC-AUC ≈ 0.8262
```

---

# 3. Run Federated Learning Simulation

## Standard FedAvg (No Differential Privacy)

```bash
flwr run . --run-config "use-dp=false" --stream
```

---

## Balanced Privacy Mode

ε = 4.0

```bash
flwr run . --stream
```

---

## Strict Privacy Mode

ε = 1.0

```bash
flwr run . --run-config "use-dp=true target-epsilon=1" --stream
```

---

# 4. Verify Secure Aggregation & Launch Dashboard

Run cryptographic verification:

```bash
python src/secure_agg.py
```

Launch Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

---

# 🧠 Architectural Insights

## Gradient Clipping Paradox in Imbalanced Medical Data

Medical datasets often contain severe class imbalance.

When DP-SGD applies:

```
max_grad_norm = 1.0
```

individual gradient contributions are clipped.

This limits the amplification introduced by class re-weighting:

```
pos_weight = 6.18
```

As a result:

- accuracy approaches majority-class prediction
- F1 score decreases
- ROC-AUC remains stable

This indicates that the model preserves ranking ability even under strong privacy constraints.

---

# Resilient Node Aggregation

Federated healthcare systems must handle unreliable networks.

During simulated failure:

- Client dropout occurs mid-training
- Server isolates failed node
- Remaining clients continue aggregation

The global model updates successfully without stopping the federation.

---

# Zero-Trust Security Model

Traditional federated learning protects raw data but still exposes model updates.

FedShield-AI adds another security layer:

- client updates are masked
- server cannot inspect individual updates
- only aggregated information is revealed

This protects against:

- inference attacks
- gradient leakage
- malicious aggregation inspection

---

# 🌐 Live Demo

Try the deployed FedShield-AI Governance Dashboard:

🔗 Demo Link: <https://ps30-federated-ai.streamlit.app/>


---

# 🛠️ Technology Stack

## Machine Learning

- Python
- PyTorch
- Scikit-learn
- NumPy
- Pandas

## Federated Learning

- Flower (`flwr`)

## Privacy

- Opacus
- Differential Privacy
- Rényi Accounting

## Visualization

- Streamlit
- Matplotlib

## Security

- Pairwise Secure Aggregation
- Cryptographic Masking

---

# 📌 Future Improvements

Possible extensions:

- Real hospital-scale deployment
- Blockchain-based audit logging
- Secure hardware enclaves
- Multi-modal clinical data support
- Real-time federated monitoring
- Kubernetes-based federation deployment

---

# 👥 Contributors

Developed as a privacy-preserving clinical AI research prototype.

---

# 📜 License

This project is released for educational and research purposes.