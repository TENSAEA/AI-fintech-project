**Name:** Tensae Aschalew
**ID:** UGR/3976/17

# AI Fintech Fraud Detection System

A comprehensive Fraud Detection System for financial services, leveraging Machine Learning (Supervised & Unsupervised) and Graph Analytics to identify suspicious transactions and fraud rings. Built with Python and Streamlit.

## 🚀 Features

*   **Dual-Model Approach**: Combines **XGBoost** (Supervised) for pattern recognition and **Isolation Forest** (Unsupervised) for anomaly detection.
*   **Graph Analytics**: Utilizes **NetworkX** to detect "Fraud Rings" and visualize complex transaction relationships.
*   **Interactive Dashboard**: A user-friendly **Streamlit** interface for real-time fraud analysis, data visualization, and explainability.
*   **Synthetic Data Generation**: Includes a robust data generator to simulate realistic transaction data with injected fraud patterns.
*   **Real-time Inference**: Simulates streaming transactions to demonstrate the system's capability in a live environment.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Frontend**: Streamlit
*   **Machine Learning**: Scikit-learn, XGBoost
*   **Graph Processing**: NetworkX
*   **Data Manipulation**: Pandas, NumPy
*   **Visualization**: Matplotlib, Plotly (via Streamlit)

## 📂 Project Structure

```
AI-fintech-project/
├── data/               # Generated transaction data
├── models/             # Saved ML models (XGBoost, Isolation Forest)
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── app.py          # Main Streamlit application
│   ├── data_generator.py # Synthetic data generation logic
│   ├── graph_utils.py  # Graph algorithms for fraud ring detection
│   └── train_models.py # Script to train and save models
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## ⚡ Getting Started

### Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/TENSAEA/AI-fintech-project.git
    cd AI-fintech-project
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Generate Data & Train Models** (Optional - models are included):
    ```bash
    python src/train_models.py
    ```

2.  **Run the Application**:
    ```bash
    streamlit run src/app.py
    ```

## 📊 How It Works

1.  **Data Ingestion**: The system ingests transaction data (amount, location, time, etc.).
2.  **Preprocessing**: Features are engineered, and data is scaled.
3.  **Model Scoring**:
    *   **XGBoost** predicts the probability of fraud based on learned patterns.
    *   **Isolation Forest** flags anomalies that deviate from normal behavior.
4.  **Graph Analysis**: Transactions are mapped as a graph (Sender -> Receiver). Cycles and dense subgraphs are identified as potential fraud rings.
5.  **Visualization**: The Streamlit dashboard presents these insights in an accessible format for analysts.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
