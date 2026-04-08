# 🎭 Sentiment Classifier with Explainability

This is a **web app** that tells you if a piece of text is **positive** or **negative**.  
It also **highlights which words** made the decision – so you can see why the AI thinks what it thinks.

No coding experience needed. Just follow the steps below.

---

## 📦 What you need

- A computer (Windows, Mac, or Linux)
- Python installed (version 3.8 to 3.11)
- An internet connection (to download the AI model once)

---

## 🚀 How to run the app on your own computer

### 1. Download or clone this repository

If you know Git:
```bash
git clone https://github.com/NhaNhaa/Sentiment_Classifier_Structure.git
cd Sentiment_Classifier_Structure
```

If you don’t know Git:  
Click the green **"Code"** button on GitHub, choose **"Download ZIP"**, then unzip the folder.

### 2. Create a virtual environment (optional but recommended)

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) inside the project folder.

- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

This will download all the libraries (Streamlit, Transformers, SHAP, etc.). It may take a few minutes.

### 4. Run the app

```bash
streamlit run app.py
```

Your default browser will open automatically with the app.

---

## 🧠 How to use the app

1. **Type or paste** any text into the big text box (e.g., a movie review, a tweet, a product comment).
2. Click the **"Analyze Sentiment"** button.
3. The app will show:
   - **Sentiment** 😊 POSITIVE or 😞 NEGATIVE
   - **Confidence** – how sure the AI is (0% to 100%)
   - (Optional) **Word importance** – words are colored **red** (pushes toward POSITIVE) or **blue** (pushes toward NEGATIVE). Hover over a word to see its exact influence number.

### Example

Try this sentence:  
*“I don’t hate this movie, it’s actually pretty good.”*

The model should say **POSITIVE** with ~75% confidence, and the words “don’t hate” and “good” will be red.

---

## ⚙️ Options (sidebar)

- **Show SHAP word explanation** – turn this on to see the colored word‑by‑word breakdown. It may take 30–60 seconds the first time you use it.

---

## 🔧 Troubleshooting

### “Streamlit is not recognized”
Make sure you activated the virtual environment (step 2) and ran `pip install -r requirements.txt`.

### The app is very slow the first time I use SHAP
That’s normal – the model is loading and SHAP is computing. The next times will be faster.

### I get an error about “text too long”
Keep your text under **2000 characters** (about 300 words). The app will warn you.

---

## 🙏 Credits

- **Model**: [Twitter‑RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) by Cardiff NLP / Hugging Face
- **Explainability**: [SHAP](https://shap.readthedocs.io/) by Scott Lundberg & Su-In Lee
- **Framework**: [Hugging Face Transformers](https://huggingface.co/docs/transformers) & [Streamlit](https://streamlit.io/)

---

## 📄 License

This project is for educational purposes. Feel free to use and modify it.

---