"""Tests for sentiment_scorer module."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import pytest
from src.sentiment_scorer import VADERScorer, SentimentScorer


def test_vader_positive():
    v = VADERScorer()
    score = v.score("Great earnings beat, stocks soar!")
    assert score > 0

def test_vader_negative():
    v = VADERScorer()
    score = v.score("Terrible loss, bankruptcy looming")
    assert score < 0

def test_vader_empty_string():
    v = VADERScorer()
    assert v.score("") == 0.0

def test_vader_batch_length():
    v = VADERScorer()
    texts = ["good news"] * 5
    out = v.score_batch(texts)
    assert len(out) == 5

def test_vader_score_range():
    v = VADERScorer()
    texts = ["awesome results", "terrible losses", "normal report"]
    scores = v.score_batch(texts)
    assert all(-1.0 <= s <= 1.0 for s in scores)

def test_scorer_no_finbert():
    scorer = SentimentScorer(vader_weight=1.0, use_finbert=False)
    df = pd.DataFrame({"headline": ["Company beats estimates", "Revenue decline reported"]})
    result = scorer.score_dataframe(df)
    assert "sentiment_score" in result.columns
    assert len(result) == 2

def test_scorer_output_columns():
    scorer = SentimentScorer(vader_weight=1.0, use_finbert=False)
    df = pd.DataFrame({"headline": ["Test headline"]})
    result = scorer.score_dataframe(df)
    for col in ["vader_score", "finbert_score", "sentiment_score"]:
        assert col in result.columns

def test_scorer_invalid_weight():
    with pytest.raises(ValueError):
        SentimentScorer(vader_weight=1.5)

def test_scorer_score_dtype():
    scorer = SentimentScorer(vader_weight=1.0, use_finbert=False)
    df = pd.DataFrame({"headline": ["Q3 results beat expectations"]})
    result = scorer.score_dataframe(df)
    assert result["sentiment_score"].dtype == np.float32
