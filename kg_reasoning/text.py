from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

import numpy as np


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\n", " ")).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = normalize_text(text).replace(" ", "")
    if not compact:
        return []
    if len(compact) < n:
        return [compact]
    return [compact[index : index + n] for index in range(len(compact) - n + 1)]


def hash_vector(text: str, dim: int = 96, ngram: int = 3) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float64)
    pieces = char_ngrams(text, n=ngram)
    if not pieces:
        return vector
    for piece in pieces:
        digest = hashlib.md5(piece.encode("utf-8")).hexdigest()
        index = int(digest, 16) % dim
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.float64)
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    safe_norm = np.where(matrix_norm == 0.0, 1.0, matrix_norm)
    return (matrix @ vector) / (safe_norm * vector_norm)


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)
