# 🎬 Sentira-Torch AI: Advanced Neural Sentiment Analysis Engine

Sentira-Torch AI is an end-to-end Deep Learning project designed to classify movie reviews into **Positive** or **Negative** sentiments. Built from scratch using **PyTorch**, the project showcases the full machine learning lifecycle—from text preprocessing and feature engineering to model optimization and deployment via an interactive **Gradio** web interface.

---

## 🚀 Project Overview & Key Features

- **Robust Text Processing:** Cleans text data using custom Tokenization and WordNet Lemmatization via NLTK. It strategically preserves negation keywords (e.g., *not, no, never, however*) within the stop-words dictionary to safeguard contextual sentiment semantics.
- **Optimized Deep Learning Architecture:** Utilizes a custom Multi-Layer Perceptron (MLP) featuring fully connected layers and Dropout layers to successfully mitigate overfitting.
- **Sleek UI Dashboard:** Wrapped in a high-fidelity, custom-themed Gradio dashboard for real-time text stream evaluation and interactive prediction visualization.

---

## 📊 Dataset & Feature Extraction

- **Dataset:** The model was trained and evaluated on the widely recognized **IMDb Dataset (50K Proportional Movie Reviews)**.
  - 👉 [Access Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Vectorization:** Texts are mapped into numerical representations using a **TF-IDF Vectorizer** constrained to the top `5,000` high-frequency features.

---

## 📈 Experiments & Performance Analysis

We conducted iterative structural experiments to unlock maximum capability from the linear feature embeddings. The final metrics are summarized below:

| Performance Metric | Experiment 1 (Baseline) | Experiment 2 (Optimized Architecture) |
| :--- | :---: | :---: |
| **Input Features** | 5,000 (TF-IDF) | 5,000 (TF-IDF) |
| **Hidden Neurons** | 32 Layers | **64 Layers** |
| **Learning Rate** | 0.001 | **0.0007** |
| **Dropout Rate** | 0.5 | 0.5 |
| **Peak Test Accuracy** | 87.42% | **87.73%** |
| **Final Test Accuracy** | 86.96% | **87.61%** |
| **Final Test Loss** | 0.3288 | **0.3019** |

### 🔍 Architectural Insights
- **Experiment 1:** Showed rapid convergence, but a noticeable gap between training accuracy and test accuracy revealed mild overfitting and memorization of training noise during later stages.
- **Experiment 2:** By scaling model capacity to 64 neurons and lowering the learning rate to 0.0007, the architecture successfully established exceptional generalization stability, driving down the final test loss to an optimized **0.3019** with higher confidence.

---

## 🛠️ Local Installation & Deployment Guide

Follow these sequential steps to set up and launch **Sentira-Torch AI** on your local workstation:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Sentira-Torch-AI.git](https://github.com/your-username/Sentira-Torch-AI.git)
cd Sentira-Torch-AI

2. Install Dependencies
Bash
pip install torch gradio nltk scikit-learn pandas numpy

3. Verify Model Artifacts
Ensure that your pre-trained pipelines are saved and located directly in the root directory:

sentiment_model.pth (PyTorch Weights)

tfidf_vectorizer.pkl (TF-IDF Mappings)

4. Execute the Application
Run the Python pipeline script to spin up the Gradio web server:

Bash
python app.py
Once launched, copy the local or public shared URL provided in the terminal to interact with the neural core interface.

