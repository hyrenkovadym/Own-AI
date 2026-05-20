from __future__ import annotations

import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9']+", flags=re.UNICODE)

# Легкі службові слова, щоб модель краще фокусувалась на змісті.
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
    # Уніфікуємо різні типи апострофів.
    t = t.replace("’", "'").replace("`", "'").replace("ʼ", "'").replace("ʹ", "'")
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


class NaiveBayesIntentModel:
    def __init__(self, alpha: float = 1.0, seed: int | None = None) -> None:
        self.alpha = alpha
        self.rng = random.Random(seed)

        self.intent_doc_counts: Counter[str] = Counter()
        self.intent_word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.intent_total_words: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.responses: dict[str, list[str]] = {}
        self.example_phrases: dict[str, str] = {}
        self.example_records: list[tuple[str, str, set[str]]] = []
        self.total_docs = 0

    def fit(self, intents: list[dict[str, object]]) -> None:
        self.intent_doc_counts.clear()
        self.intent_word_counts = defaultdict(Counter)
        self.intent_total_words.clear()
        self.vocab.clear()
        self.responses = {}
        self.example_phrases = {}
        self.example_records = []
        self.total_docs = 0

        for intent_entry in intents:
            name = str(intent_entry["name"])
            examples = [str(x) for x in intent_entry.get("examples", [])]
            responses = [str(x) for x in intent_entry.get("responses", [])]
            self.responses[name] = responses

            for text in examples:
                tokens = tokenize(text)
                if not tokens:
                    continue
                phrase = normalize_phrase(text)
                if phrase and phrase not in self.example_phrases:
                    self.example_phrases[phrase] = name
                self.example_records.append((phrase, name, set(tokens)))
                self.total_docs += 1
                self.intent_doc_counts[name] += 1
                self.intent_word_counts[name].update(tokens)
                self.intent_total_words[name] += len(tokens)
                self.vocab.update(tokens)

        if self.total_docs == 0:
            raise ValueError("У датасеті не знайдено прикладів для навчання.")

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

        # 1) Fuzzy-матч до прикладів: дає кращу стійкість до опечаток.
        fuzzy_intent, fuzzy_conf, fuzzy_known_ratio = self._fuzzy_match(phrase=phrase, tokens=tokens)
        if fuzzy_intent is not None and fuzzy_conf >= 0.86:
            return Prediction(
                intent=fuzzy_intent,
                confidence=fuzzy_conf,
                margin=max(0.0, fuzzy_conf - 0.55),
                known_token_ratio=fuzzy_known_ratio,
                token_count=len(raw_tokens),
            )

        if not tokens:
            return Prediction(
                intent="fallback",
                confidence=0.0,
                margin=0.0,
                known_token_ratio=0.0,
                token_count=len(raw_tokens),
            )

        known_tokens = [t for t in tokens if t in self.vocab]
        known_ratio = (len(known_tokens) / len(raw_tokens)) if raw_tokens else 0.0

        # Якщо є середній fuzzy і майже невідомі слова, краще взяти fuzzy-клас.
        if fuzzy_intent is not None and fuzzy_conf >= 0.70 and known_ratio < 0.25:
            return Prediction(
                intent=fuzzy_intent,
                confidence=min(0.92, fuzzy_conf),
                margin=max(0.0, fuzzy_conf - 0.50),
                known_token_ratio=max(known_ratio, fuzzy_known_ratio),
                token_count=len(raw_tokens),
            )

        if not known_tokens:
            return Prediction(
                intent="fallback",
                confidence=0.0,
                margin=0.0,
                known_token_ratio=0.0,
                token_count=len(raw_tokens),
            )

        intents = list(self.intent_doc_counts.keys())
        if not intents:
            return Prediction(
                intent="fallback",
                confidence=0.0,
                margin=0.0,
                known_token_ratio=0.0,
                token_count=len(raw_tokens),
            )

        # 2) Класичний Naive Bayes по токенах.
        vocab_size = max(1, len(self.vocab))
        scores: dict[str, float] = {}
        for intent in intents:
            prior = math.log(self.intent_doc_counts[intent] / self.total_docs)
            score = prior

            total_words = self.intent_total_words[intent]
            word_counts = self.intent_word_counts[intent]
            for token in known_tokens:
                token_count = word_counts[token]
                prob = (token_count + self.alpha) / (total_words + self.alpha * vocab_size)
                score += math.log(prob)
            scores[intent] = score

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_items[0]

        shifted = {k: math.exp(v - best_score) for k, v in scores.items()}
        denom = sum(shifted.values())
        probs = {k: (val / denom if denom > 0 else 0.0) for k, val in shifted.items()}
        best_conf = probs.get(best_intent, 0.0)

        second_conf = 0.0
        if len(sorted_items) > 1:
            second_intent = sorted_items[1][0]
            second_conf = probs.get(second_intent, 0.0)

        # Легка поправка на fuzzy, якщо він збігається з NB.
        if fuzzy_intent == best_intent and fuzzy_conf >= 0.60:
            best_conf = min(0.98, 0.82 * best_conf + 0.18 * fuzzy_conf)

        return Prediction(
            intent=best_intent,
            confidence=best_conf,
            margin=max(0.0, best_conf - second_conf),
            known_token_ratio=known_ratio,
            token_count=len(raw_tokens),
        )

    def _fuzzy_match(self, phrase: str, tokens: list[str]) -> tuple[str | None, float, float]:
        if not phrase or not self.example_records:
            return None, 0.0, 0.0

        token_set = set(tokens)
        best_intent: str | None = None
        best_score = 0.0
        best_known_ratio = 0.0

        for ex_phrase, ex_intent, ex_tokens in self.example_records:
            char_sim = SequenceMatcher(None, phrase, ex_phrase).ratio()
            if not token_set and not ex_tokens:
                token_sim = 1.0
            else:
                union = token_set | ex_tokens
                token_sim = (len(token_set & ex_tokens) / len(union)) if union else 0.0

            # Символи + токени разом.
            score = 0.62 * char_sim + 0.38 * token_sim
            if score > best_score:
                best_score = score
                best_intent = ex_intent
                best_known_ratio = token_sim

        return best_intent, best_score, best_known_ratio

    def choose_response(self, intent: str) -> str:
        options = self.responses.get(intent, [])
        if not options:
            return "Я ще не знаю. Спробуй перефразувати або навчи мене через /teach."
        return self.rng.choice(options)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "intent_doc_counts": dict(self.intent_doc_counts),
            "intent_word_counts": {k: dict(v) for k, v in self.intent_word_counts.items()},
            "intent_total_words": dict(self.intent_total_words),
            "vocab": list(self.vocab),
            "responses": self.responses,
            "example_phrases": self.example_phrases,
            "example_records": [(p, i, list(t)) for p, i, t in self.example_records],
            "total_docs": self.total_docs,
        }
        with path.open("wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayesIntentModel":
        with Path(path).open("rb") as f:
            data = pickle.load(f)

        model = cls(alpha=data.get("alpha", 1.0))
        model.intent_doc_counts = Counter(data.get("intent_doc_counts", {}))
        model.intent_word_counts = defaultdict(Counter)
        for k, v in data.get("intent_word_counts", {}).items():
            model.intent_word_counts[k] = Counter(v)
        model.intent_total_words = Counter(data.get("intent_total_words", {}))
        model.vocab = set(data.get("vocab", []))
        model.responses = data.get("responses", {})
        model.example_phrases = data.get("example_phrases", {})
        raw_records = data.get("example_records", [])
        model.example_records = [(p, i, set(t)) for p, i, t in raw_records]
        model.total_docs = int(data.get("total_docs", 0))
        return model
