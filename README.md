# 🎬 NeuralSentiment-PyTorch

## 1. Problem Description
This project addresses the challenge of **Sentiment Analysis** in movie reviews. The goal is to build an automated Neural Network system that can process raw text and accurately classify it as **Positive** or **Negative**. This is a binary classification problem that requires handling complex linguistic features like negations, context, and vocabulary variations.

## 2. Dataset Link
The model was trained using the **IMDb Dataset (50K Movie Reviews)**.
👉 [Click here to view Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## 3. Results (Comparison Table)
We conducted two experiments to reach the optimal configuration. The results are as follows:

| Feature | Experiment 1 (Baseline) | Experiment 2 (Optimized) |
| :--- | :--- | :--- |
| **Input Features** | 5,000 (TF-IDF) | 5,000 (TF-IDF) |
| **Hidden Neurons** | 32 | **64** |
| **Learning Rate** | 0.001 | **0.0007** |
| **Test Accuracy** | 87.57% | **87.60%** |
| **Test Loss** | 0.3071 | **0.2979** |

*Experiment 2 achieved better stability and a lower loss rate by increasing model capacity and refining the learning rate.*

## 4. Instructions for Running the Project
Follow these steps to run the application on your local machine:
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/NeuralSentiment-PyTorch.git](https://github.com/your-username/NeuralSentiment-PyTorch.git)
   cd NeuralSentiment-PyTorch
2. **Install the required libraries:**
   ```bash
       pip install streamlit torch nltk scikit-learn

3. **Ensure model files are present:**
   Make sure `sentiment_model.pth` and `tfidf_vectorizer.pkl` are in the main directory.

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py

   
