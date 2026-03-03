"""
LangfairMetricsCalculator - computes similarity metrics between
model-generated answers and reference/gold text.

Produces:
 cosine similarity over TF-IDF embeddings
 ROUGE-L score (coverage of longest common subsequence)
 BLEU score (n-gram precision)

These metrics provide complementary perspectives on answer quality.
"""

from __future__ import annotations
from typing import Dict, Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class LangfairMetricsCalculator:
    def __init__(self):
        self.smoothing = SmoothingFunction()
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def compute(self, pred: str, ref: str):
        reference_tokens = nltk.word_tokenize(ref.lower())
        prediction_tokens = nltk.word_tokenize(pred.lower())

        bleu = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            smoothing_function=self.smoothing.method1,
        )

        rougeL = self.rouge.score(ref, pred)['rougeL'].fmeasure

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([ref, pred])
        cosine = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        return {
            "cosine": float(cosine),
            "rougeL": float(rougeL),
            "bleu": float(bleu),
        }

