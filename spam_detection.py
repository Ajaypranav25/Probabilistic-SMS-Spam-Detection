"""Probabilistic SMS Spam Detection (script version of spam_detection.ipynb).

Run:  python spam_detection.py
"""


# # Probabilistic SMS Spam Detection
# **Course:** BAMAT207 Probability and Statistics - Project Activity
# **Topic:** Probability and Random Variables
#
# This notebook classifies an SMS message as **Spam** or **Ham** (genuine) using:
# dataset analysis, conditional probability, Bayes' theorem, expectation and variance of a random variable,
# and a Multinomial Naive Bayes classifier.

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

DATA_FILE = Path("data/SMSSpamCollection")
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
SUSPICIOUS_WORDS = ["free", "prize", "win", "urgent", "offer"]

# ## 1. Dataset Handling
# Dataset: **SMS Spam Collection** (UCI Machine Learning Repository, 5,574 labelled messages).
#
# Note: `ucimlrepo.fetch_ucirepo(id=228)` reports that this dataset is *not available for import*,
# so the official UCI zip file is downloaded directly instead (and kept in `data/` so it also works offline).

def load_dataset():
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(DATA_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        archive.extract("SMSSpamCollection", DATA_FILE.parent)
    # The file is tab separated: label <TAB> message. quoting=3 keeps quote characters as plain text.
    return pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "message"], quoting=3)


df = load_dataset()
print("Messages loaded:", len(df))
print(df.head())

print("Missing values per column:")
print(df.isnull().sum())

rows_before = len(df)
df = df.drop_duplicates(subset="message").reset_index(drop=True)
print("\nDuplicate messages removed:", rows_before - len(df))
print("Messages after cleaning:", len(df))

# Spam = 1, Ham = 0
df["label_num"] = df["label"].map({"spam": 1, "ham": 0})
print(df.head())

# ## 2. Dataset Analysis
# P(Spam) = spam messages / total messages, P(Ham) = ham messages / total messages

total = len(df)
n_spam = int((df["label_num"] == 1).sum())
n_ham = int((df["label_num"] == 0).sum())

p_spam = n_spam / total
p_ham = n_ham / total

print("Cleaned dataset summary")
print("Total messages :", total)
print("Spam messages  :", n_spam)
print("Ham messages   :", n_ham)
print(f"P(Spam) = {n_spam}/{total} = {p_spam:.4f}")
print(f"P(Ham)  = {n_ham}/{total} = {p_ham:.4f}")
print("Check: P(Spam) + P(Ham) =", round(p_spam + p_ham, 4))

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Ham (0)", "Spam (1)"], [n_ham, n_spam], color=["#bbbbbb", "#333333"], edgecolor="black")
ax.bar_label(bars, labels=[f"{n_ham} ({p_ham:.1%})", f"{n_spam} ({p_spam:.1%})"])
ax.set_title("Class Distribution of SMS Messages")
ax.set_ylabel("Number of messages")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
plt.close()

# ## 3. Word-Based Probability Analysis
# For each suspicious word W:
#
# - Direct count: P(Spam | W) = spam messages containing W / all messages containing W
# - Bayes' theorem: P(Spam | W) = P(W | Spam) x P(Spam) / P(W)
#
# Both methods must give the same answer. A word is matched as a whole word, ignoring case.

def contains_word(messages, word):
    pattern = rf"\b{re.escape(word)}\b"
    return messages.str.contains(pattern, case=False, regex=True)


rows = []
for word in SUSPICIOUS_WORDS:
    has_word = contains_word(df["message"], word)
    n_word = int(has_word.sum())
    n_spam_word = int((has_word & (df["label_num"] == 1)).sum())

    direct = n_spam_word / n_word
    p_word_given_spam = n_spam_word / n_spam
    p_word = n_word / total
    bayes = p_word_given_spam * p_spam / p_word

    rows.append({
        "word": word,
        "messages with word": n_word,
        "spam with word": n_spam_word,
        "P(W|Spam)": round(p_word_given_spam, 4),
        "P(W)": round(p_word, 4),
        "P(Spam|W) direct": round(direct, 4),
        "P(Spam|W) Bayes": round(bayes, 4),
    })

word_table = pd.DataFrame(rows)
print(word_table)

# Worked example for the word "free"
free = word_table.set_index("word").loc["free"]
print("Direct  : P(Spam|free) =", int(free["spam with word"]), "/", int(free["messages with word"]),
      "=", free["P(Spam|W) direct"])
print("Bayes   : P(Spam|free) = P(free|Spam) x P(Spam) / P(free)")
print(f"          = {free['P(W|Spam)']} x {p_spam:.4f} / {free['P(W)']} = {free['P(Spam|W) Bayes']}")

# ## 4. Random Variable Analysis
# Y = 1 if the message is spam, Y = 0 if it is ham. Y is a **Bernoulli** random variable with p = P(Spam).
#
# - E(Y) = 1 x p + 0 x (1 - p) = p
# - Var(Y) = E(Y^2) - [E(Y)]^2 = p - p^2 = p(1 - p)

Y = df["label_num"].to_numpy()
p = p_spam

expected_formula = p
variance_formula = p * (1 - p)

print(f"E(Y)   formula  = p          = {expected_formula:.4f}")
print(f"E(Y)   numpy    = mean(Y)    = {np.mean(Y):.4f}")
print(f"Var(Y) formula  = p(1 - p)   = {variance_formula:.4f}")
print(f"Var(Y) numpy    = var(Y)     = {np.var(Y):.4f}")
print(f"Std(Y)          = sqrt(Var)  = {np.sqrt(variance_formula):.4f}")

# ## 5. Data Preparation
# 80% training / 20% testing with **stratified** sampling, so both parts keep the same spam-ham ratio.
# `CountVectorizer` turns each message into word counts. It is fitted on the training data only.

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["message"], df["label_num"],
    test_size=TEST_SIZE, stratify=df["label_num"], random_state=RANDOM_STATE,
)

print("Training messages:", len(X_train_text), f"(spam share {y_train.mean():.4f})")
print("Testing messages :", len(X_test_text), f"(spam share {y_test.mean():.4f})")

vectorizer = CountVectorizer(lowercase=True)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print("Vocabulary size (number of word features):", len(vectorizer.vocabulary_))

# ## 6. Model Building
# Multinomial Naive Bayes uses Bayes' theorem with the "naive" assumption that words are independent given the class.
# It works well with word counts.

model = MultinomialNB()
model.fit(X_train, y_train)

print("Prior P(Ham), P(Spam) learned from training data:", np.round(np.exp(model.class_log_prior_), 4))

# ## 7. Model Evaluation
# Accuracy, Precision, Recall, F1-score and the Confusion Matrix on the 20% test set.

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")
report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], digits=4)
print(report)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)
print(f"\nSpam messages incorrectly classified as genuine (False Negatives): {fn}")
print(f"Genuine messages incorrectly classified as spam (False Positives): {fp}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Greys")
labels = ["Ham", "Spam"]
ax.set_xticks([0, 1], labels=labels)
ax.set_yticks([0, 1], labels=labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_title("Confusion Matrix (Test Set)")
for i in range(2):
    for j in range(2):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color=text_color)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

test_results = pd.DataFrame({"message": X_test_text, "actual": y_test, "predicted": y_pred})

print("Spam predicted as Ham (missed spam):")
for msg in test_results.query("actual == 1 and predicted == 0")["message"].head(5):
    print(" -", msg)

print("\nHam predicted as Spam (false alarm):")
for msg in test_results.query("actual == 0 and predicted == 1")["message"].head(5):
    print(" -", msg)

# ## 8. Real-Time Testing
# `predict_message` returns the predicted class and the probabilities P(Spam | Message) and P(Ham | Message).

def predict_message(message):
    features = vectorizer.transform([message])
    p_ham_msg, p_spam_msg = model.predict_proba(features)[0]
    prediction = "Spam" if p_spam_msg >= 0.5 else "Ham (Genuine)"
    print("Message:", message)
    print("Prediction:", prediction)
    print(f"P(Spam | Message) = {p_spam_msg:.4f}")
    print(f"P(Ham | Message) = {p_ham_msg:.4f}\n")
    return prediction, p_spam_msg, p_ham_msg


predict_message("Congratulations! You have won a free prize. Claim now.")
predict_message("Hey, are we still meeting for lunch at 1 today?")
predict_message("URGENT! Your mobile number has been selected for a cash offer. Call now.")

# Type your own SMS here and run this cell
my_message = "Reminder: your library book is due tomorrow."
predict_message(my_message)

# Save a short text summary of all results
with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as f:
    f.write("Probabilistic SMS Spam Detection - Results\n\n")
    f.write(f"Messages after cleaning: {total} (spam {n_spam}, ham {n_ham})\n")
    f.write(f"P(Spam) = {p_spam:.4f}, P(Ham) = {p_ham:.4f}\n")
    f.write(f"E(Y) = {expected_formula:.4f}, Var(Y) = {variance_formula:.4f}\n\n")
    f.write(word_table.to_string(index=False) + "\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n{report}\n")
    f.write(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}\n")
print("Saved outputs/results.txt")


if __name__ == "__main__":
    print("\nType an SMS to classify it (press Enter on an empty line to quit).")
    while True:
        user_message = input("SMS: ").strip()
        if not user_message:
            break
        predict_message(user_message)
