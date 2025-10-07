# Twitter Sentiment Analysis

A machine learning project that classifies tweets as positive or negative using the Sentiment140 dataset.

## Overview

Built and compared three different classification models to analyze sentiment in tweets. The goal was to see how well traditional ML approaches could handle social media text, which is known for being messy with slang, abbreviations, and emojis.

## Dataset

- **Source**: Sentiment140 dataset from Kaggle
- **Size**: 1.6 million tweets
- **Classes**: Binary (0 = negative, 4 = positive)
- **Split**: 80/20 train-test split

The dataset was perfectly balanced with 800k tweets per class after filtering.

## Approach

### Text Processing
- Basic text cleaning (lowercase conversion)
- TF-IDF vectorization with unigrams and bigrams
- Limited to 5000 features to keep training manageable

### Models Tested
1. **Bernoulli Naive Bayes** - Fast baseline classifier
2. **Linear SVM** - Good for high-dimensional text data
3. **Logistic Regression** - Simple and interpretable

## Results

| Model | Accuracy | F1-Score | Notes |
|-------|----------|----------|-------|
| BernoulliNB | 76.6% | 0.77 | Fastest but least accurate |
| SVM | 79.5% | 0.80 | Great balance of speed and accuracy |
| Logistic Regression | 79.6% | 0.80 | Best overall performance |

Both Logistic Regression and SVM performed nearly identically, correctly classifying about 255,000 out of 320,000 test tweets.

## What I Learned

- TF-IDF with bigrams works surprisingly well for sentiment classification
- Simple models can achieve decent results (~80%) on large datasets without complex preprocessing
- The performance gap between Naive Bayes and logistic models shows the value of testing multiple approaches
- All models had similar recall for both classes, suggesting no major bias toward positive or negative predictions

## Room for Improvement

- Better text cleaning (removing URLs, mentions, special characters)
- Handling negations properly ("not good" vs "good")
- Testing with different n-gram ranges
- Trying ensemble methods
- Using word embeddings or transformer models for comparison

## Requirements

```
pandas
scikit-learn
matplotlib
seaborn
```

## Usage

1. Download the Sentiment140 dataset from Kaggle
2. Place it in the project directory as `training.1600000.processed.noemoticon.csv`
3. Run the notebook

The trained Logistic Regression model can classify new tweets with about 80% accuracy, good enough for basic sentiment monitoring tasks.
