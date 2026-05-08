# Tasks 60–62 — NLP: Linguistics, Text Processing & Text Analysis
# Libraries: NLTK, re, collections

import re
import string
from collections import Counter
import nltk

# Download required NLTK data
nltk.download("punkt",         quiet=True)
nltk.download("punkt_tab",     quiet=True)
nltk.download("stopwords",     quiet=True)
nltk.download("wordnet",       quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("vader_lexicon",  quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ═══════════════════════════════════════════════════════
# Task 60 — Linguistics using Machine Learning / NLP
# ═══════════════════════════════════════════════════════
print("=" * 55)
print("Task 60 — Linguistics using NLP in Python")
print("=" * 55)

print("""
── What is NLP? ──
Natural Language Processing (NLP) is a branch of AI that
enables computers to understand, interpret, and generate
human language. Core linguistic tasks include:
  - Tokenisation  : splitting text into words/sentences
  - POS Tagging   : labelling each word's grammatical role
  - NER           : identifying named entities (people, places)
  - Parsing       : understanding sentence structure
""")

text = "Artificial Intelligence is transforming industries worldwide. Pakistan has a growing tech sector with talented engineers in cities like Lahore and Karachi."

# Sentence tokenisation
sentences = sent_tokenize(text)
print("── Sentence Tokenisation ──")
for i, s in enumerate(sentences, 1):
    print(f"  Sentence {i}: {s}")

# Word tokenisation
words = word_tokenize(text)
print(f"\n── Word Tokenisation ──")
print(f"  Total words : {len(words)}")
print(f"  Tokens      : {words[:15]}...")

# POS Tagging
print(f"\n── Part-of-Speech (POS) Tagging ──")
pos_tags = pos_tag(words)
pos_legend = {
    "NNP": "Proper Noun", "NN": "Noun", "VBZ": "Verb",
    "JJ": "Adjective", "IN": "Preposition", "DT": "Determiner",
    "VBG": "Verb (gerund)", "NNS": "Plural Noun"
}
for word, tag in pos_tags[:12]:
    desc = pos_legend.get(tag, tag)
    print(f"  {word:<20} → {tag:<6} ({desc})")


# ═══════════════════════════════════════════════════════
# Task 61 — Text Processing with NLTK
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("Task 61 — Text Processing with NLTK")
print("=" * 55)

sample = """
Machine learning is a subset of artificial intelligence.
It enables computers to learn from data without being explicitly programmed.
Deep learning uses neural networks with many layers to learn complex patterns.
Natural language processing helps computers understand human language.
"""

print("Original Text:")
print(sample.strip())

# Step 1: Lowercase
text_lower = sample.lower()
print("\n── Step 1: Lowercase ──")
print(text_lower.strip()[:100] + "...")

# Step 2: Remove punctuation
text_clean = text_lower.translate(str.maketrans("", "", string.punctuation))
print("\n── Step 2: Remove Punctuation ──")
print(text_clean.strip()[:100] + "...")

# Step 3: Tokenise
tokens = word_tokenize(text_clean)
print(f"\n── Step 3: Tokenise → {len(tokens)} tokens ──")
print(tokens[:15])

# Step 4: Remove stopwords
stop_words = set(stopwords.words("english"))
filtered = [w for w in tokens if w not in stop_words and w.isalpha()]
print(f"\n── Step 4: Remove Stopwords → {len(filtered)} tokens remaining ──")
print(filtered[:15])

# Step 5: Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
print(f"\n── Step 5: Stemming (reduces words to root form) ──")
for orig, stem in list(zip(filtered, stemmed))[:8]:
    print(f"  {orig:<20} → {stem}")

# Step 6: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
print(f"\n── Step 6: Lemmatization (dictionary base form) ──")
for orig, lem in list(zip(filtered, lemmatized))[:8]:
    print(f"  {orig:<20} → {lem}")

# Step 7: Word frequency
freq = Counter(filtered)
print(f"\n── Step 7: Word Frequency (Top 10) ──")
for word, count in freq.most_common(10):
    print(f"  {word:<20}: {count}")


# ═══════════════════════════════════════════════════════
# Task 62 — Text Analysis
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("Task 62 — Text Analysis")
print("=" * 55)

reviews = [
    "This product is absolutely amazing! I love it so much.",
    "Terrible experience. The product broke after one day. Very disappointed.",
    "It is okay. Not the best, not the worst. Average quality.",
    "Fantastic quality and fast delivery! Highly recommend to everyone.",
    "Worst purchase ever. Complete waste of money. Never buying again.",
]

# Sentiment Analysis with VADER
print("── Sentiment Analysis with VADER ──")
sia = SentimentIntensityAnalyzer()
for review in reviews:
    scores = sia.polarity_scores(review)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "POSITIVE 😊"
    elif compound <= -0.05:
        sentiment = "NEGATIVE 😠"
    else:
        sentiment = "NEUTRAL  😐"
    print(f"\n  Text     : {review[:55]}...")
    print(f"  Scores   : pos={scores['pos']:.2f} neg={scores['neg']:.2f} neu={scores['neu']:.2f}")
    print(f"  Compound : {compound:.3f} → {sentiment}")

# Text statistics
all_text = " ".join(reviews)
all_words = word_tokenize(all_text.lower())
all_words = [w for w in all_words if w.isalpha()]

print(f"\n── Text Statistics Across All Reviews ──")
print(f"  Total words        : {len(all_words)}")
print(f"  Unique words       : {len(set(all_words))}")
print(f"  Vocabulary richness: {len(set(all_words)) / len(all_words):.2%}")

# Top keywords (excluding stopwords)
stop_words = set(stopwords.words("english"))
keywords = [w for w in all_words if w not in stop_words]
top_keywords = Counter(keywords).most_common(8)
print(f"\n  Top Keywords:")
for word, count in top_keywords:
    bar = "█" * count
    print(f"  {word:<15}: {bar} ({count})")

# N-gram analysis (bigrams)
bigrams = [(all_words[i], all_words[i+1]) for i in range(len(all_words)-1)]
bigram_freq = Counter(bigrams).most_common(5)
print(f"\n  Top Bigrams (word pairs):")
for bigram, count in bigram_freq:
    print(f"  {' '.join(bigram):<25}: {count}")

print("\n── Pipeline Summary ──")
print("  Raw Text → Tokenise → Clean → Stopword Removal")
print("  → Stem/Lemmatize → Frequency Analysis → Sentiment")