# Urdu Binary Sentiment Classifier

This repository contains the `sentiment-analyzer.ipynb` notebook from the Sarcasm Detector project, which implements a binary Urdu sentiment classifier using a small curated dataset of labeled Urdu sentences.

## Project overview

This repository contains:

- a labeled Urdu dataset
- a stopword list for preprocessing
- a Jupyter notebook that trains and evaluates a deep-learning model
- a saved TensorFlow/Keras model file
- sample labeled sentences for reference

The workflow in the notebook is:

1. load the Urdu dataset
2. lowercase and clean text
3. remove Urdu stopwords and punctuation
4. tokenize and pad sentences
5. train a bidirectional LSTM model
6. save the trained model as a `.keras` file
7. evaluate and run predictions

## Repository contents

| File | Purpose |
| --- | --- |
| `sentiment-analyzer.ipynb` | Main notebook for preprocessing, training, evaluation, and saving the model |
| `Urdu_dataset.csv` | Primary labeled dataset with Urdu sentences and binary labels |
| `stopwords.txt` | Urdu stopwords removed during preprocessing |
| `samples_copy.txt` | Example labeled Urdu sentences used as quick references/samples |
| `Urdu_Sarcasm_Detector.keras` | Saved Keras model produced by the notebook |
| `LICENSE` | MIT license for the project |
| `.gitignore` | Ignored files and local-only artifacts |

## Dataset

`Urdu_dataset.csv` contains 50 labeled Urdu sentences with the columns:

- `Sentence`
- `Sentiment`
- `Id`

The dataset is balanced, with 25 examples in each class.

## Model pipeline

The notebook uses the following configuration:

- sequence length: `15`
- vocabulary size: `10000`
- embedding dimension: `16`
- model: `Embedding -> Bidirectional LSTM -> Bidirectional LSTM -> Dense`
- optimizer: `Adam`
- loss: `binary_crossentropy`
- training split: `90%` train / `10%` test

## How to run

### Requirements

- Python 3.9+
- TensorFlow
- pandas
- numpy
- matplotlib
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install tensorflow pandas numpy matplotlib jupyter
```

### Train and evaluate

1. Open `sentiment-analyzer.ipynb`
2. Run the cells from top to bottom
3. The trained model will be saved as `Urdu_Sarcasm_Detector.keras`
4. Accuracy and loss plots will be displayed during evaluation

### Use the saved model

After training, the saved model can be loaded in TensorFlow and used for inference on new Urdu text.

## Notes

- This project is notebook-driven rather than packaged as a standalone app.
- The dataset is small, so results should be treated as experimental.
- The notebook's 90/10 split leaves only 5 test samples, so evaluation is limited and cross-validation would be a better option for a stronger estimate.
- The notebook currently reports predictions as `Positive Sentiment` or `Negative Sentiment` based on the model output, so the saved model behaves as a binary sentiment classifier.
- The `Sentiment` column uses binary labels, and the notebook evaluates the model as positive vs negative sentiment.

## License

This project is released under the MIT License. See `LICENSE` for details.
