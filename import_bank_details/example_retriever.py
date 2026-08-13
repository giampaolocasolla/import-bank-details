"""KATE-style kNN retrieval of few-shot examples by merchant-name similarity."""

from collections import Counter
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from import_bank_details.expense_names import clean_expense_name

_HARD_CAP = 100
_NGRAM_SIZES = (3, 4, 5)


def _input_text(fields: dict[str, Any]) -> str:
    """Build retrieval text from an expense input: cleaned name plus optional comment."""
    cleaned = clean_expense_name(str(fields.get("Expense_name", "") or ""))
    comment = str(fields.get("Comment", "") or "").strip()
    if comment:
        return f"{cleaned} {comment}"
    return cleaned


def _char_wb_ngrams(text: str) -> list[str]:
    """Return character n-grams (n in {3, 4, 5}) with word-boundary padding."""
    grams: list[str] = []
    for word in text.lower().split():
        padded = f" {word} "
        length = len(padded)
        for n in _NGRAM_SIZES:
            if length >= n:
                grams.extend(padded[i : i + n] for i in range(length - n + 1))
    return grams


def _fit_tfidf(
    documents: list[str],
) -> tuple[dict[str, int], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit a character n-gram TF-IDF matrix and L2-normalize each row."""
    n_docs = len(documents)
    doc_ngrams = [_char_wb_ngrams(doc) for doc in documents]

    vocab: dict[str, int] = {}
    for grams in doc_ngrams:
        for gram in grams:
            if gram not in vocab:
                vocab[gram] = len(vocab)

    n_terms = len(vocab)
    if n_docs == 0 or n_terms == 0:
        return vocab, np.zeros(n_terms, dtype=np.float64), np.zeros((n_docs, n_terms), dtype=np.float64)

    df = np.zeros(n_terms, dtype=np.float64)
    for grams in doc_ngrams:
        for gram in set(grams):
            df[vocab[gram]] += 1.0

    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0

    matrix = np.zeros((n_docs, n_terms), dtype=np.float64)
    for row, grams in enumerate(doc_ngrams):
        if not grams:
            continue
        for gram, tf in Counter(grams).items():
            matrix[row, vocab[gram]] = float(tf) * idf[vocab[gram]]

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)
    matrix /= row_norms
    return vocab, idf, matrix


class ExampleRetriever:
    """Select the most similar labeled examples for a batch of expenses.

    Indexes examples once using character n-gram TF-IDF (n in {3, 4, 5},
    word-boundary padded) and cosine similarity. The instance is immutable
    after init so it can be shared across threads.
    """

    def __init__(self, examples: list[dict[str, Any]], max_examples: int = 32) -> None:
        self._examples: tuple[dict[str, Any], ...] = tuple(examples)
        self._max_examples = max(0, min(_HARD_CAP, max_examples))
        self._vocab, self._idf, self._matrix = _fit_tfidf([_input_text(example["input"]) for example in self._examples])

    def retrieve_for_batch(self, expenses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return up to ``max_examples`` examples most similar to any expense in the batch.

        Small pools (``len(examples) <= max_examples``) are returned in original order.
        Otherwise examples are ranked by max cosine similarity across the batch, dropping
        zero-score matches. Results are ordered least similar first (ICL recency bias).
        """
        if self._max_examples == 0 or not self._examples:
            return []
        if len(self._examples) <= self._max_examples:
            return list(self._examples)

        merged = np.zeros(len(self._examples), dtype=np.float64)
        for expense in expenses:
            merged = np.maximum(merged, self._cosine_scores(expense))

        if not np.any(merged > 0):
            return []

        candidate_idx = np.flatnonzero(merged > 0)
        k = min(self._max_examples, candidate_idx.size)
        candidate_scores = merged[candidate_idx]
        top_local = np.argpartition(candidate_scores, -k)[-k:]
        top_idx = candidate_idx[top_local]
        order = np.argsort(merged[top_idx], kind="stable")
        return [self._examples[int(i)] for i in top_idx[order]]

    def _cosine_scores(self, expense: dict[str, Any]) -> npt.NDArray[np.float64]:
        """Return cosine similarity of one query against every indexed example."""
        vec = self._query_vector(_input_text(expense))
        if self._matrix.shape[1] == 0:
            return np.zeros(self._matrix.shape[0], dtype=np.float64)
        return cast(npt.NDArray[np.float64], np.einsum("ij,j->i", self._matrix, vec))

    def _query_vector(self, text: str) -> npt.NDArray[np.float64]:
        n_terms = self._idf.size
        vec = np.zeros(n_terms, dtype=np.float64)
        if n_terms == 0:
            return vec
        grams = [gram for gram in _char_wb_ngrams(text) if gram in self._vocab]
        if not grams:
            return vec
        for gram, tf in Counter(grams).items():
            vec[self._vocab[gram]] = float(tf) * self._idf[self._vocab[gram]]
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec
