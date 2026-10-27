# Probabilistic SMS Spam Detection

BAMAT207 Probability and Statistics - Project Activity (Topic: Probability and Random Variables)

A simple Python system that classifies an SMS message as **Spam** or **Ham** (genuine). It analyses the
SMS Spam Collection dataset, calculates probabilities (conditional probability, Bayes' theorem,
expectation and variance), trains a Multinomial Naive Bayes model and shows the prediction probability.

## Files

| File | What it is |
|------|------------|
| `spam_detection.ipynb` | Main notebook with all 8 activities and their outputs |
| `spam_detection.py` | Same code as a plain script, with a prompt to type your own SMS |
| `data/SMSSpamCollection` | Dataset (UCI, 5,574 labelled messages) |
| `outputs/class_distribution.png` | Spam vs ham bar chart |
| `outputs/confusion_matrix.png` | Confusion matrix of the test set |
| `outputs/results.txt` | Summary of all probabilities and model scores |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook spam_detection.ipynb   # notebook version
python spam_detection.py                # script version
```

The notebook also runs on Google Colab (upload it together with the `data` folder, or let it download the dataset).

## Results (after removing duplicates: 5,171 messages)

| Quantity | Value |
|----------|-------|
| Spam / Ham messages | 653 / 4,518 |
| P(Spam), P(Ham) | 0.1263, 0.8737 |
| E(Y), Var(Y) | 0.1263, 0.1103 |

| Word | Messages with word | Spam with word | P(Spam \| word) |
|------|------|------|------|
| free | 203 | 148 | 0.7291 |
| prize | 73 | 73 | 1.0000 |
| win | 55 | 49 | 0.8909 |
| urgent | 62 | 57 | 0.9194 |
| offer | 30 | 24 | 0.8000 |

The direct count and Bayes' theorem give the same value for every word.

| Metric (test set, 1,035 messages) | Value |
|--------|-------|
| Accuracy | 0.9807 |
| Spam precision / recall / F1 | 0.9512 / 0.8931 / 0.9213 |
| Missed spam (spam predicted as ham) | 14 |
| False alarms (ham predicted as spam) | 6 |

## Dataset

Almeida, T.A., Gomez Hidalgo, J.M. (2011). SMS Spam Collection. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
(`ucimlrepo` cannot import this dataset, so the official zip is downloaded directly.)
