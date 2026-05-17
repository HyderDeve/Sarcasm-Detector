# Sarcasm Detector

A small Urdu sarcasm/sentiment detection project built around a TensorFlow BiLSTM model and a Streamlit demo app.

## What is in this repository?

- `sentiment-analyzer.ipynb` — end-to-end notebook for preprocessing Urdu text, training the model, evaluating it, and saving the final weights.
- `web_app.py` — Streamlit interface for running predictions from the saved model.
- `Urdu_dataset.csv` — labeled training dataset with `Sentence`, `Sentiment`, and `Id` columns.
- `stopwords.txt` — Urdu stopwords used during preprocessing.
- `Urdu_Sarcasm_Detector.keras` — trained Keras model artifact used by the app.
- `samples_copy.txt` — example Urdu sentences with labels for quick reference.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT license.
- `.gitignore` — ignores helper/reference files and a dataset zip.

## Project overview

The notebook trains a binary text classifier on Urdu sentences using:

- text cleaning with stopword removal and punctuation handling
- tokenization with a vocabulary size of 10,000
- padding to a maximum sequence length of 15
- a Bidirectional LSTM model with an embedding layer and dense classifier

The Streamlit app loads `Urdu_Sarcasm_Detector.keras`, accepts an Urdu sentence, applies the same preprocessing flow, and returns a binary prediction.

## Dataset notes

- `Urdu_dataset.csv` contains 50 labeled examples.
- Labels in the repository use `1` for negative and `0` for positive, which is an inverted encoding compared with some common sentiment-analysis conventions.
- `samples_copy.txt` mirrors the same style of examples and can be used to understand the dataset format.

## Preprocessing

The project removes Urdu punctuation and a custom stopword list before tokenization.
The notebook and app both rely on the same general preprocessing idea, so new data should be prepared consistently.

## Model details

- Vocabulary size: `10000`
- Sequence length: `15`
- Embedding dimension: `16`
- Architecture: Embedding → Bidirectional LSTM → Bidirectional LSTM → Dense → Sigmoid
- Saved model: `Urdu_Sarcasm_Detector.keras`

## Running the Streamlit app

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run web_app.py
   ```
3. Enter an Urdu sentence and click **Check Sentiment**.

## Retraining the model

Open `sentiment-analyzer.ipynb` and run the cells in order. The notebook:

1. loads `Urdu_dataset.csv`
2. cleans and tokenizes the text
3. trains the model
4. evaluates the model
5. saves the trained artifact as `Urdu_Sarcasm_Detector.keras`

## Notes

- The project is intended as a lightweight demo and works best with the same preprocessing assumptions used in the notebook.
- If you update the training pipeline, retrain the model and refresh the saved `.keras` file so the Streamlit app stays in sync.

## License

This project is released under the MIT License. See `LICENSE` for details.
