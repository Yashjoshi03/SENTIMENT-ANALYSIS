# SENTIMENT-ANALYSIS
COMPANY: CODTECH IT SOLUTIONS

NAME: YASH JOSHI

INTERN ID: CT12DL193

DOMAIN: DATA ANALYTICS

DURATION: 12 WEEKS

MENTOR: NEELA SANTOSH

# 📊 Sentiment Analysis Using NLP and Logistic Regression

This project performs sentiment analysis on textual data (tweets) using Natural Language Processing (NLP) techniques and machine learning with logistic regression. The workflow includes data preprocessing, TF-IDF vectorization, model training, and evaluation.

## ⚙️ Technologies & Libraries

* **Language**: Python
* **Libraries**:

  * `pandas`, `numpy` – Data manipulation
  * `matplotlib`, `seaborn` – Data visualization
  * `nltk` – Natural Language Toolkit for text preprocessing
  * `scikit-learn` – ML model and evaluation metrics

---

## 🚀 Workflow Overview

### 1. **Data Preprocessing**

* Dropped missing text rows.
* Text cleaning: lowercase, punctuation/URL removal, tokenization, stopword removal, and stemming using NLTK.

### 2. **Exploratory Data Analysis (EDA)**

* Visualized sentiment distribution using seaborn countplot.

### 3. **Text Vectorization**

* Applied `TF-IDF` with max 5000 features to convert text into numeric format.

### 4. **Model Development**

* Used `Logistic Regression` from scikit-learn.
* Labels encoded with `LabelEncoder`.
* Data split into training and testing sets (80/20).

### 5. **Model Evaluation**

* Metrics: Accuracy Score, Classification Report (Precision, Recall, F1-score), and Confusion Matrix.

### 6. **Inference**

* Tested model on new sample tweets and displayed predicted sentiments.

---

## 📈 Results

* **Model**: Logistic Regression
* **Performance Metrics**: Printed classification report and visual confusion matrix
* **Sample Predictions**:

  * "I love this game!" → Positive
  * "Worst experience ever" → Negative
  * "Just okay." → Neutral
Output:
<img width="558" height="393" alt="Image" src="https://github.com/user-attachments/assets/0db39e60-7cc0-49a4-b810-6bf51abcceb1" />

<img width="563" height="240" alt="Image" src="https://github.com/user-attachments/assets/320365dd-6a51-4016-a0be-15145ff40e86" />

<img width="518" height="393" alt="Image" src="https://github.com/user-attachments/assets/50011f19-5323-4167-99b2-4c29a3bb3b36" />

<img width="572" height="68" alt="Image" src="https://github.com/user-attachments/assets/46e3d826-e74c-467a-ae89-a456f818ca54" />

## ✅ Future Enhancements

* Use more advanced models like Random Forest, SVM, or Deep Learning (LSTM/BERT).
* Implement multi-language support.
* Expand dataset and handle sarcasm/irony detection.

