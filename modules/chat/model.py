from __future__ import annotations

import math
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9']+", flags=re.UNICODE)

STOPWORDS = {
    "і",
    "й",
    "та",
    "або",
    "але",
    "бо",
    "що",
    "це",
    "цей",
    "ця",
    "ці",
    "той",
    "такий",
    "так",
    "ні",
    "як",
    "коли",
    "де",
    "хто",
    "який",
    "яка",
    "яке",
    "які",
    "мене",
    "мені",
    "тобі",
    "тобою",
    "ми",
    "ви",
    "вони",
    "він",
    "вона",
    "воно",
    "у",
    "в",
    "на",
    "до",
    "з",
    "із",
    "за",
    "по",
    "a",
    "an",
    "the",
    "is",
    "are",
    "am",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "it",
    "this",
    "that",
    "i",
    "you",
    "me",
    "my",
    "your",
    "we",
    "they",
    "he",
    "she",
    "what",
    "who",
    "how",
}


def _normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = t.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    return t


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(_normalize_text(text))]
    if not drop_stopwords:
        return tokens
    filtered = [t for t in tokens if t not in STOPWORDS]
    return filtered or tokens


def normalize_phrase(text: str) -> str:
    return " ".join(tokenize(text, drop_stopwords=False))


@dataclass
class Prediction:
    intent: str
    confidence: float
    margin: float = 0.0
    known_token_ratio: float = 0.0
    token_count: int = 0


class NeuralIntentModel:
    """A tiny MLP intent classifier trained from local examples.

    Architecture:
    - bag-of-words input
    - hidden tanh layer
    - softmax output

    No external API and no heavy deps required.
    """

    MODEL_TYPE = "mlp_intent_v1"

    def __init__(
        self,
        hidden_size: int = 24,
        learning_rate: float = 0.06,
        epochs: int = 240,
        l2: float = 1e-5,
        seed: int | None = None,
    ) -> None:
        self.hidden_size = max(4, int(hidden_size))
        self.learning_rate = float(learning_rate)
        self.epochs = max(20, int(epochs))
        self.l2 = max(0.0, float(l2))
        self.rng = random.Random(seed)

        self.responses: dict[str, list[str]] = {}
        self.example_phrases: dict[str, str] = {}
        self.vocab: list[str] = []
        self.vocab_index: dict[str, int] = {}
        self.token_vocab: set[str] = set()
        self.intent_to_idx: dict[str, int] = {}
        self.idx_to_intent: list[str] = []

        self.W1: list[list[float]] = []
        self.b1: list[float] = []
        self.W2: list[list[float]] = []
        self.b2: list[float] = []

    def fit(self, intents: list[dict[str, object]]) -> None:
        self.responses = {}
        self.example_phrases = {}

        docs: list[tuple[list[str], str]] = []
        vocab_set: set[str] = set()
        token_vocab_set: set[str] = set()

        for intent_entry in intents:
            intent_name = str(intent_entry.get("name", "")).strip()
            if not intent_name:
                continue

            examples = [str(x) for x in intent_entry.get("examples", [])]
            responses = [str(x) for x in intent_entry.get("responses", [])]
            if responses:
                self.responses[intent_name] = responses

            for text in examples:
                phrase = normalize_phrase(text)
                if phrase and phrase not in self.example_phrases:
                    self.example_phrases[phrase] = intent_name

                tokens = tokenize(text)
                if not tokens:
                    continue
                token_vocab_set.update(tokens)

                features = self._extract_features(tokens=tokens, phrase=phrase)
                docs.append((features, intent_name))
                vocab_set.update(features)

        if not docs:
            raise ValueError("У датасеті не знайдено прикладів для навчання.")

        self.vocab = sorted(vocab_set)
        self.vocab_index = {tok: i for i, tok in enumerate(self.vocab)}
        self.token_vocab = token_vocab_set

        intents_sorted = sorted({intent for _, intent in docs})
        self.idx_to_intent = intents_sorted
        self.intent_to_idx = {intent: i for i, intent in enumerate(intents_sorted)}

        X_sparse: list[list[tuple[int, float]]] = []
        y: list[int] = []

        for features, intent_name in docs:
            sparse = self._vectorize_sparse(features)
            if not sparse:
                continue
            X_sparse.append(sparse)
            y.append(self.intent_to_idx[intent_name])

        if not X_sparse:
            raise ValueError("Після токенізації не лишилось даних для навчання.")

        self._init_weights(input_size=len(self.vocab), num_classes=len(self.idx_to_intent))
        self._train_sgd(X_sparse, y)

    def predict(self, text: str) -> Prediction:
        raw_tokens = tokenize(text, drop_stopwords=False)
        tokens = tokenize(text)
        if not raw_tokens:
            return Prediction(intent="fallback", confidence=0.0, margin=0.0, known_token_ratio=0.0, token_count=0)

        phrase = normalize_phrase(text)
        exact_intent = self.example_phrases.get(phrase)
        if exact_intent is not None:
            return Prediction(
                intent=exact_intent,
                confidence=0.99,
                margin=0.99,
                known_token_ratio=1.0,
                token_count=len(raw_tokens),
            )

        if not self.W1 or not self.W2:
            return Prediction(intent="fallback", confidence=0.0, margin=0.0, known_token_ratio=0.0, token_count=len(raw_tokens))

        known_tokens = [t for t in tokens if t in self.token_vocab]
        known_ratio = (len(known_tokens) / len(raw_tokens)) if raw_tokens else 0.0

        if not known_tokens:
            return Prediction(intent="fallback", confidence=0.0, margin=0.0, known_token_ratio=0.0, token_count=len(raw_tokens))

        features = self._extract_features(tokens=tokens, phrase=phrase)
        known_features = [f for f in features if f in self.vocab_index]
        if not known_features:
            return Prediction(intent="fallback", confidence=0.0, margin=0.0, known_token_ratio=known_ratio, token_count=len(raw_tokens))

        x_sparse = self._vectorize_sparse(known_features)
        probs = self._forward_probs(x_sparse)
        if not probs:
            return Prediction(intent="fallback", confidence=0.0, margin=0.0, known_token_ratio=known_ratio, token_count=len(raw_tokens))

        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_conf = probs[best_idx]
        sorted_probs = sorted(probs, reverse=True)
        second_conf = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        return Prediction(
            intent=self.idx_to_intent[best_idx],
            confidence=best_conf,
            margin=max(0.0, best_conf - second_conf),
            known_token_ratio=known_ratio,
            token_count=len(raw_tokens),
        )

    def choose_response(self, intent: str) -> str:
        options = self.responses.get(intent, [])
        if not options:
            return "Я ще не знаю. Спробуй перефразувати або навчи мене через /teach."
        base = self.rng.choice(options)

        # Tiny response composer: creates richer replies without external LLM.
        addons = self._response_addons().get(intent, [])
        if addons and self.rng.random() < 0.45:
            addon = self.rng.choice(addons)
            if addon not in base:
                return f"{base} {addon}"
        return base

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_type": self.MODEL_TYPE,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "responses": self.responses,
            "example_phrases": self.example_phrases,
            "vocab": self.vocab,
            "token_vocab": sorted(self.token_vocab),
            "idx_to_intent": self.idx_to_intent,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

        with p.open("wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralIntentModel":
        with Path(path).open("rb") as f:
            data = pickle.load(f)

        model_type = data.get("model_type") if isinstance(data, dict) else None
        if model_type != cls.MODEL_TYPE:
            if isinstance(data, dict) and "intent_doc_counts" in data:
                raise ValueError(
                    "Знайдено модель старого формату (Naive Bayes). "
                    "Натисни 'Навчити', щоб створити нейромережеву модель у новому форматі."
                )
            raise ValueError("Невідомий формат моделі. Перенавчи модель кнопкою 'Навчити'.")

        model = cls(
            hidden_size=int(data.get("hidden_size", 24)),
            learning_rate=float(data.get("learning_rate", 0.06)),
            epochs=int(data.get("epochs", 240)),
            l2=float(data.get("l2", 1e-5)),
        )
        model.responses = data.get("responses", {})
        model.example_phrases = data.get("example_phrases", {})
        model.vocab = list(data.get("vocab", []))
        model.vocab_index = {tok: i for i, tok in enumerate(model.vocab)}
        model.token_vocab = set(data.get("token_vocab", model.vocab))
        model.idx_to_intent = list(data.get("idx_to_intent", []))
        model.intent_to_idx = {intent: i for i, intent in enumerate(model.idx_to_intent)}
        model.W1 = [list(row) for row in data.get("W1", [])]
        model.b1 = list(data.get("b1", []))
        model.W2 = [list(row) for row in data.get("W2", [])]
        model.b2 = list(data.get("b2", []))
        return model

    def _init_weights(self, input_size: int, num_classes: int) -> None:
        scale1 = 1.0 / math.sqrt(max(1, input_size))
        scale2 = 1.0 / math.sqrt(max(1, self.hidden_size))

        self.W1 = [
            [(self.rng.random() * 2.0 - 1.0) * scale1 for _ in range(self.hidden_size)]
            for _ in range(input_size)
        ]
        self.b1 = [0.0 for _ in range(self.hidden_size)]

        self.W2 = [
            [(self.rng.random() * 2.0 - 1.0) * scale2 for _ in range(num_classes)]
            for _ in range(self.hidden_size)
        ]
        self.b2 = [0.0 for _ in range(num_classes)]

    def _vectorize_sparse(self, tokens: list[str]) -> list[tuple[int, float]]:
        counts: dict[int, float] = {}
        for tok in tokens:
            idx = self.vocab_index.get(tok)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0.0) + 1.0

        if not counts:
            return []

        length = float(len(tokens)) if tokens else 1.0
        return sorted((idx, val / length) for idx, val in counts.items())

    @staticmethod
    def _extract_features(tokens: list[str], phrase: str) -> list[str]:
        features: list[str] = []
        features.extend(f"w:{t}" for t in tokens)

        # Token bigrams help with short phrase meaning.
        for i in range(len(tokens) - 1):
            features.append(f"bg:{tokens[i]}_{tokens[i + 1]}")

        compact = phrase.replace(" ", "")
        if len(compact) >= 3:
            # Char trigrams help with typos and inflections.
            for i in range(len(compact) - 2):
                tri = compact[i : i + 3]
                if " " in tri:
                    continue
                features.append(f"cg:{tri}")

        return features

    @staticmethod
    def _response_addons() -> dict[str, list[str]]:
        return {
            "greeting": ["Радий бачити тебе в чаті."],
            "thanks": ["Звертайся, якщо хочеш ще прокачати модель."],
            "capabilities": ["Можу донавчатися прямо на твоїх прикладах."],
            "why": ["Якщо хочеш, можу пояснити це крок за кроком."],
        }

    def _forward_hidden(self, x_sparse: list[tuple[int, float]]) -> list[float]:
        z1 = self.b1.copy()
        for idx, val in x_sparse:
            row = self.W1[idx]
            for j in range(self.hidden_size):
                z1[j] += val * row[j]
        return [math.tanh(v) for v in z1]

    def _forward_probs(self, x_sparse: list[tuple[int, float]]) -> list[float]:
        h = self._forward_hidden(x_sparse)

        z2 = self.b2.copy()
        for j in range(self.hidden_size):
            hj = h[j]
            row = self.W2[j]
            for k in range(len(z2)):
                z2[k] += hj * row[k]

        m = max(z2)
        exps = [math.exp(v - m) for v in z2]
        denom = sum(exps)
        if denom <= 0.0:
            return [0.0 for _ in z2]
        return [e / denom for e in exps]

    def _train_sgd(self, X_sparse: list[list[tuple[int, float]]], y: list[int]) -> None:
        n = len(X_sparse)
        class_count = len(self.idx_to_intent)

        indices = list(range(n))
        for _ in range(self.epochs):
            self.rng.shuffle(indices)
            for sample_idx in indices:
                x = X_sparse[sample_idx]
                target = y[sample_idx]

                h = self._forward_hidden(x)

                z2 = self.b2.copy()
                for j in range(self.hidden_size):
                    hj = h[j]
                    row = self.W2[j]
                    for k in range(class_count):
                        z2[k] += hj * row[k]

                m = max(z2)
                exps = [math.exp(v - m) for v in z2]
                denom = sum(exps)
                probs = [e / denom for e in exps]

                delta2 = probs
                delta2[target] -= 1.0

                delta1 = [0.0 for _ in range(self.hidden_size)]
                for j in range(self.hidden_size):
                    back = 0.0
                    row = self.W2[j]
                    for k in range(class_count):
                        back += row[k] * delta2[k]
                    delta1[j] = (1.0 - h[j] * h[j]) * back

                lr = self.learning_rate

                for j in range(self.hidden_size):
                    row = self.W2[j]
                    hj = h[j]
                    for k in range(class_count):
                        grad = hj * delta2[k] + self.l2 * row[k]
                        row[k] -= lr * grad

                for k in range(class_count):
                    self.b2[k] -= lr * delta2[k]

                for input_idx, xval in x:
                    row = self.W1[input_idx]
                    for j in range(self.hidden_size):
                        grad = xval * delta1[j] + self.l2 * row[j]
                        row[j] -= lr * grad

                for j in range(self.hidden_size):
                    self.b1[j] -= lr * delta1[j]


# Backward-compatible name used by entry.py
NaiveBayesIntentModel = NeuralIntentModel
