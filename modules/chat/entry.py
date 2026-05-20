from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .model import NaiveBayesIntentModel


@dataclass
class ChatTrainStats:
    intents: int
    examples: int
    vocab_size: int
    model_path: str


@dataclass
class ChatReply:
    intent: str
    confidence: float
    text: str


class ChatModule:
    name = "chat"

    def __init__(self) -> None:
        self._kernel = None
        self._lock = RLock()
        self._model_cache: dict[str, tuple[float, NaiveBayesIntentModel]] = {}
        self._sessions: dict[str, deque[str]] = {}
        self._memory_turns = 5

    def on_load(self, kernel) -> None:
        self._kernel = kernel

    def on_unload(self) -> None:
        with self._lock:
            self._model_cache.clear()
            self._sessions.clear()
        self._kernel = None

    def train(
        self,
        dataset_path: str = "modules/chat/data/intents.json",
        model_path: str = "models/chat_intent.pkl",
        seed: int | None = None,
    ) -> ChatTrainStats:
        data = self._load_dataset(dataset_path)
        intents = data.get("intents", [])
        model = NaiveBayesIntentModel(seed=seed)
        model.fit(intents)
        model.save(model_path)
        self._put_model_in_cache(model_path, model)

        examples_count = 0
        for item in intents:
            examples_count += len(item.get("examples", []))

        return ChatTrainStats(
            intents=len(intents),
            examples=examples_count,
            vocab_size=len(model.vocab),
            model_path=str(Path(model_path)),
        )

    def reply(
        self,
        user_text: str,
        model_path: str = "models/chat_intent.pkl",
        confidence_threshold: float = 0.22,
        session_id: str = "default",
    ) -> ChatReply:
        model = self._get_model(model_path)
        pred = model.predict(user_text)

        # Якщо впевненість низька, пробуємо контекст поточної сесії.
        if pred.confidence < confidence_threshold and pred.known_token_ratio > 0.0:
            context_text = self._build_context_text(session_id, user_text)
            if context_text != user_text:
                contextual = model.predict(context_text)
                if contextual.confidence > pred.confidence:
                    pred = contextual

        if self._should_fallback(pred, confidence_threshold):
            self._remember(session_id, user_text)
            return ChatReply(
                intent="fallback",
                confidence=pred.confidence,
                text=(
                    "Я поки не впевнений у відповіді. "
                    "Спробуй перефразувати або додай приклад через форму навчання."
                ),
            )

        text = model.choose_response(pred.intent)
        self._remember(session_id, user_text)
        return ChatReply(intent=pred.intent, confidence=pred.confidence, text=text)

    def teach(
        self,
        intent: str,
        example: str,
        response: str,
        dataset_path: str = "modules/chat/data/intents.json",
    ) -> None:
        data = self._load_dataset(dataset_path)
        intents = data.setdefault("intents", [])
        intent_name = intent.strip().lower()
        example = example.strip()
        response = response.strip()

        if not intent_name or not example or not response:
            raise ValueError("Інтент, приклад і відповідь не можуть бути порожніми.")

        target = None
        for item in intents:
            if str(item.get("name", "")).lower() == intent_name:
                target = item
                break

        if target is None:
            target = {"name": intent_name, "examples": [], "responses": []}
            intents.append(target)

        examples = target.setdefault("examples", [])
        responses = target.setdefault("responses", [])
        if example not in examples:
            examples.append(example)
        if response not in responses:
            responses.append(response)

        self._save_dataset(dataset_path, data)

    def reset_session(self, session_id: str = "default") -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _remember(self, session_id: str, user_text: str) -> None:
        with self._lock:
            session = self._sessions.setdefault(session_id, deque(maxlen=self._memory_turns))
            session.append(user_text)

    def _build_context_text(self, session_id: str, user_text: str) -> str:
        with self._lock:
            session = list(self._sessions.get(session_id, []))
        if not session:
            return user_text
        return " ".join(session + [user_text])

    @staticmethod
    def _should_fallback(pred, confidence_threshold: float) -> bool:
        # Сильна впевненість: приймаємо відповідь.
        if pred.confidence >= max(0.72, confidence_threshold + 0.12):
            return False

        # Для коротких фраз із адекватним покриттям не форсуємо fallback.
        if pred.confidence >= confidence_threshold and pred.token_count <= 4 and pred.known_token_ratio >= 0.45:
            return False

        if pred.confidence >= confidence_threshold + 0.08 and pred.token_count <= 4:
            return False

        if pred.confidence < confidence_threshold:
            return True

        # Дуже низьке покриття словника + слабка впевненість.
        if pred.token_count >= 5 and pred.known_token_ratio < 0.30 and pred.confidence < 0.65:
            return True

        # Невеликий відрив від 2-го місця теж сигнал невпевненості.
        if pred.margin < 0.03 and pred.confidence < 0.60:
            return True

        return False

    def _put_model_in_cache(self, model_path: str, model: NaiveBayesIntentModel) -> None:
        path = Path(model_path)
        resolved = str(path.resolve())
        mtime = path.stat().st_mtime
        with self._lock:
            self._model_cache[resolved] = (mtime, model)

    def _get_model(self, model_path: str) -> NaiveBayesIntentModel:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл моделі не знайдено: {model_path}")
        resolved = str(path.resolve())
        mtime = path.stat().st_mtime

        with self._lock:
            cached = self._model_cache.get(resolved)
            if cached is not None and cached[0] == mtime:
                return cached[1]

        model = NaiveBayesIntentModel.load(path)
        with self._lock:
            self._model_cache[resolved] = (mtime, model)
        return model

    @staticmethod
    def _load_dataset(dataset_path: str) -> dict:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл датасету не знайдено: {dataset_path}")
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    @staticmethod
    def _save_dataset(dataset_path: str, data: dict) -> None:
        path = Path(dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")


def create_module() -> ChatModule:
    return ChatModule()
