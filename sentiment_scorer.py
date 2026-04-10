"""
sentiment_scorer.py
-------------------
Dual-engine sentiment scoring pipeline combining VADER and FinBERT.

Author : EquityXperts / Tasrif
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class VADERScorer:
    """Thin wrapper around VADER SentimentIntensityAnalyzer."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER scorer initialised.")

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return float(self._analyzer.polarity_scores(text)["compound"])

    def score_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.score(t) for t in texts], dtype=np.float32)


class FinBERTScorer:
    """
    Wrapper around ProsusAI/finbert.
    Score = P(positive) - P(negative) in [-1, +1].
    Lazy-loads on first call.
    """

    MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 512
    BATCH_SIZE = 16

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            device_id = 0 if torch.cuda.is_available() else -1
            if self._device is not None:
                device_id = self._device

            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                device=device_id,
                top_k=None,
                truncation=True,
                max_length=self.MAX_LENGTH,
            )
            logger.info("FinBERT loaded on device=%s.", device_id)
        except Exception as exc:
            logger.warning("FinBERT load failed: %s. Returning zeros.", exc)
            self._pipeline = None

    def _raw_to_score(self, output: list[dict]) -> float:
        label_scores = {item["label"].lower(): item["score"] for item in output}
        return float(label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0))

    def score_batch(self, texts: list[str]) -> np.ndarray:
        self._load()
        if self._pipeline is None:
            return np.zeros(len(texts), dtype=np.float32)

        scores: list[float] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i: i + self.BATCH_SIZE]
            try:
                results = self._pipeline(batch)
                for res in results:
                    scores.append(self._raw_to_score(res))
            except Exception as exc:
                logger.warning("FinBERT batch error at index %d: %s", i, exc)
                scores.extend([0.0] * len(batch))

        return np.array(scores, dtype=np.float32)


class SentimentScorer:
    """
    Ensemble scorer: blended VADER + FinBERT sentiment signal.

    Parameters
    ----------
    vader_weight : float
        Weight for VADER; FinBERT weight = 1 - vader_weight.
    use_finbert : bool
        If False, uses VADER only (much faster, no download).
    """

    def __init__(
        self,
        vader_weight: float = 0.4,
        use_finbert: bool = True,
        device: Optional[str] = None,
    ) -> None:
        if not 0.0 <= vader_weight <= 1.0:
            raise ValueError("vader_weight must be in [0, 1].")
        self.vader_weight = vader_weight
        self.finbert_weight = 1.0 - vader_weight
        self._vader = VADERScorer()
        self._finbert: Optional[FinBERTScorer] = (
            FinBERTScorer(device=device) if use_finbert else None
        )

    def score_dataframe(
        self, df: pd.DataFrame, headline_col: str = "headline"
    ) -> pd.DataFrame:
        """
        Score all headlines and return enriched DataFrame.

        New columns: vader_score, finbert_score, sentiment_score.
        """
        texts = df[headline_col].tolist()
        vader_scores = self._vader.score_batch(texts)

        if self._finbert is not None:
            finbert_scores = self._finbert.score_batch(texts)
        else:
            finbert_scores = np.zeros(len(texts), dtype=np.float32)

        blended = self.vader_weight * vader_scores + self.finbert_weight * finbert_scores

        out = df.copy()
        out["vader_score"] = vader_scores
        out["finbert_score"] = finbert_scores
        out["sentiment_score"] = blended.astype(np.float32)
        return out
