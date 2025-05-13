## HealthSense: Disease Forecasting 🧠🩺

**HealthSense** is an AI‑powered web application that predicts multiple diseases based on patient data and symptoms. The system integrates classical Machine Learning, Deep Learning, and a Reinforcement Learning prototype, delivering fast, accurate forecasts and interactive visualizations via a Streamlit interface.

👉 **[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/HassanAnees/HealthSense-Forecasting)**

---

## 🚀 Key Features

- **Multi‑Disease Support**: Diabetes, Heart Disease, Lung Cancer, Kidney Disease, Hypertension, Breast Cancer, and more.
- **Machine Learning Models**: Decision Tree, Random Forest, SVM, KNN, Logistic Regression.
- **Deep Learning Models**: Feedforward ANN for tabular data; CNN for imaging data (e.g., lung scans).
- **Reinforcement Learning Prototype**: DQN agent simulating adaptive treatment strategies.
- **Interactive Visualizations**: Probability bars, correlation heatmaps, feature distributions, and Plotly charts.
- **User‑Friendly Interface**: Streamlit-based web app for symptom input, live predictions, and result exploration.
- **Modular Codebase**: Clean separation of data, models, notebooks, and application code.

---

## 🧰 Tools & Platforms Used

### Programming & ML Stack
- Python, NumPy, Pandas, Scikit-learn, XGBoost
- TensorFlow/Keras for Deep Learning
- OpenAI Gym & Stable-Baselines3 for RL

### Visualization & UI
- Streamlit (Web App)
- Matplotlib, Seaborn, Plotly (Charts)
- FastAPI (Optional API Backend)

### Development Tools
- Jupyter Notebooks, Google Colab
- Git & GitHub
- VS Code

### Hosting & Demos
- Hugging Face Spaces – Live Demo
- Kaggle Datasets – Source Data

---

## 📦 Project Structure

```

HealthSense-Disease-Forecasting/
├── app/
│   └── app.py                 # Streamlit application
├── data/
│   ├── raw/                   # Downloaded Kaggle CSV files
│   └── processed/             # Cleaned and feature‑engineered data
├── models/
│   ├── ml/                    # Serialized ML models (.pkl)
│   ├── dl/                    # Saved DL models (.h5)
│   └── rl/                    # RL agent (DQN) checkpoints
├── notebooks/                 # EDA, ML, DL, and RL experiments
├── src/
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train\_\*.py             # Training scripts for ML/DL/RL
│   └── visualizer.py          # Plot generation
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and instructions

```

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HassanAnees1/HealthSense-Disease-Forecasting.git
   cd HealthSense-Disease-Forecasting
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Activate:
   source venv/bin/activate       # Unix/macOS
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Kaggle datasets into `data/raw/`:**

   * diabetes.csv
   * heart.csv
   * lung\_cancer.csv
   * kidney.csv
   * hypertension.csv
   * breast\_cancer.csv

   ### Dataset Links

   * [Diabetes (Pima Indians)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
   * [Heart Disease](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
   * [Lung Cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
   * [Chronic Kidney Disease](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
   * [Hypertension Risk](https://www.kaggle.com/datasets/khan1803115/hypertension-risk-model-main)
   * [Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

5. **Preprocess and train models**

   ```bash
   python src/preprocess.py
   python src/train_diabetes.py
   python src/train_heart.py
   python src/train_lung_cancer.py
   python src/train_kidney.py
   python src/train_hypertension.py
   python src/train_breast_cancer.py
   python src/train_dl.py
   python src/train_rl.py
   ```

6. **Run the Streamlit app locally**

   ```bash
   streamlit run app/app.py
   ```

---

## 👨‍💻 Team & Contacts

* **Hassan Anees** – Project Lead, DL & RL, Visualization
  [LinkedIn](https://www.linkedin.com/in/hassananees)

* **Adham \[Last Name]** – Data Engineering & ML
  [LinkedIn](https://www.linkedin.com/in/adham_profile)

* **Kirols \[Last Name]** – Backend, UI Integration
  [LinkedIn](https://www.linkedin.com/in/kirols_profile)

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full details.

---

> Made with ❤️ by Team HealthSense – [Try it on Hugging Face](https://huggingface.co/spaces/HassanAnees/HealthSense-Forecasting)

---
