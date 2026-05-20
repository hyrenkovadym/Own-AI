from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from .local_llm import OllamaClient
from .model import NeuralIntentModel


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
    DEFAULT_FALLBACK_QUEUE_PATH = "modules/chat/data/fallback_queue.json"
    DEFAULT_CODER_MODEL = "qwen2.5-coder:7b"

    def __init__(self) -> None:
        self._kernel = None
        self._lock = RLock()
        self._model_cache: dict[str, tuple[float, NeuralIntentModel]] = {}
        self._sessions: dict[str, deque[str]] = {}
        self._session_bot_names: dict[str, str] = {}
        self._memory_turns = 5
        self._ollama = OllamaClient()

    def on_load(self, kernel) -> None:
        self._kernel = kernel

    def on_unload(self) -> None:
        with self._lock:
            self._model_cache.clear()
            self._sessions.clear()
            self._session_bot_names.clear()
        self._kernel = None

    def train(
        self,
        dataset_path: str = "modules/chat/data/intents.json",
        model_path: str = "models/chat_intent.pkl",
        seed: int | None = None,
    ) -> ChatTrainStats:
        data = self._load_dataset(dataset_path)
        intents = data.get("intents", [])
        model = NeuralIntentModel(seed=seed)
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
        rename_target = self._extract_custom_name(user_text)
        if rename_target is not None:
            with self._lock:
                self._session_bot_names[session_id] = rename_target
            self._remember(session_id, user_text)
            return ChatReply(
                intent="set_name",
                confidence=1.0,
                text=f"Домовились, можеш називати мене {rename_target}.",
            )

        if self._is_name_question(user_text):
            custom_name = self._get_custom_name(session_id)
            if custom_name is not None:
                self._remember(session_id, user_text)
                return ChatReply(
                    intent="name",
                    confidence=1.0,
                    text=f"Я {custom_name}.",
                )

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
            self._record_fallback(user_text=user_text, session_id=session_id)
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
        if pred.intent == "name":
            custom_name = self._get_custom_name(session_id)
            if custom_name is not None:
                text = f"Я {custom_name}."
        self._remember(session_id, user_text)
        return ChatReply(intent=pred.intent, confidence=pred.confidence, text=text)

    def reply_programmer(
        self,
        user_text: str,
        model_path: str = "models/chat_intent.pkl",
        confidence_threshold: float = 0.22,
        session_id: str = "default",
        coder_model: str = DEFAULT_CODER_MODEL,
    ) -> ChatReply:
        text = user_text.strip()
        if not text:
            return ChatReply(intent="fallback", confidence=0.0, text="Опиши задачу, і я почну роботу.")

        rename_target = self._extract_custom_name(text)
        if rename_target is not None:
            with self._lock:
                self._session_bot_names[session_id] = rename_target
            self._remember(session_id, user_text)
            return ChatReply(
                intent="set_name",
                confidence=1.0,
                text=f"Домовились, можеш називати мене {rename_target}.",
            )

        if self._is_name_question(text):
            custom_name = self._get_custom_name(session_id)
            if custom_name is not None:
                self._remember(session_id, text)
                return ChatReply(intent="name", confidence=1.0, text=f"Я {custom_name}.")

        if self._ollama.is_available():
            system_prompt = (
                "Ти локальний асистент-програміст у desktop-проєкті користувача. "
                "Відповідай українською, коротко і практично. "
                "Для запитів по коду спершу дай міні-план (1-4 кроки), потім робочий код/команди. "
                "Не вигадуй факти про файли, якщо їх не бачиш."
            )
            llm = self._ollama.generate(
                prompt=text,
                model=(coder_model or self.DEFAULT_CODER_MODEL).strip(),
                system=system_prompt,
                temperature=0.15,
            )
            if llm.ok:
                self._remember(session_id, text)
                return ChatReply(intent="programmer", confidence=0.99, text=llm.text)

        fallback = self.reply(
            user_text=text,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            session_id=session_id,
        )
        if fallback.intent == "fallback":
            msg = (
                "Я ще не маю локальної coding-моделі або вона вимкнена. "
                "Щоб отримати режим програміста, запусти Ollama і встанови модель "
                f"`{self.DEFAULT_CODER_MODEL}`. Потім пиши задачу прямо в чат."
            )
            return ChatReply(intent="programmer_unavailable", confidence=0.0, text=msg)
        return fallback

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

    def fallback_queue_stats(
        self,
        queue_path: str = DEFAULT_FALLBACK_QUEUE_PATH,
    ) -> tuple[int, int]:
        queue = self._load_fallback_queue(queue_path)
        items = queue.get("items", [])
        unique_items = len(items)
        total_count = sum(int(item.get("count", 0)) for item in items)
        return unique_items, total_count

    def fallback_peek(
        self,
        queue_path: str = DEFAULT_FALLBACK_QUEUE_PATH,
    ) -> str | None:
        queue = self._load_fallback_queue(queue_path)
        items = queue.get("items", [])
        if not items:
            return None
        sorted_items = sorted(
            items,
            key=lambda x: (int(x.get("count", 0)), str(x.get("last_seen", ""))),
            reverse=True,
        )
        text = str(sorted_items[0].get("text", "")).strip()
        return text or None

    def fallback_consume(
        self,
        text: str,
        queue_path: str = DEFAULT_FALLBACK_QUEUE_PATH,
        count: int = 1,
    ) -> bool:
        target = text.strip()
        if not target:
            return False
        dec = max(1, int(count))

        queue = self._load_fallback_queue(queue_path)
        items = queue.get("items", [])
        changed = False

        new_items = []
        for item in items:
            item_text = str(item.get("text", "")).strip()
            item_count = int(item.get("count", 0))
            if item_text != target:
                new_items.append(item)
                continue
            changed = True
            remain = item_count - dec
            if remain > 0:
                item["count"] = remain
                new_items.append(item)

        if changed:
            queue["items"] = new_items
            self._save_fallback_queue(queue_path, queue)
        return changed

    def reset_session(self, session_id: str = "default") -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._session_bot_names.pop(session_id, None)

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

    def _get_custom_name(self, session_id: str) -> str | None:
        with self._lock:
            return self._session_bot_names.get(session_id)

    @staticmethod
    def _is_name_question(text: str) -> bool:
        low = text.strip().lower()
        markers = (
            "як тебе звати",
            "як твоє ім",
            "твоє ім",
            "хто ти",
            "ти хто",
            "як тебе називати",
        )
        return any(m in low for m in markers)

    @staticmethod
    def _extract_custom_name(text: str) -> str | None:
        raw = text.strip()
        if not raw:
            return None

        patterns = (
            r"(?:давай(?:\s+тебе)?(?:\s+буде)?\s+звати)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"(?:пропоную\s+тебе\s+назвати)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"(?:тепер\s+тебе\s+звати)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"(?:ми\s+тебе(?:\s+ж)?\s+називали)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"(?:називали\s+тебе)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"(?:я\s+)?називатиму\s+тебе\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"(?:називатиму\s+тебе)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"(?:будеш\s+мати\s+ім(?:'|’|`)?я)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})",
            r"^(?:ні[,\s]+)?ти\s+([A-Za-zА-Яа-яІіЇїЄєҐґ][A-Za-zА-Яа-яІіЇїЄєҐґ0-9'_-]{1,31})$",
        )

        for pattern in patterns:
            m = re.search(pattern, raw, flags=re.IGNORECASE)
            if m is None:
                continue
            candidate = m.group(1).strip(" .,!?:;\"'`()[]{}")
            if not candidate:
                continue
            low = candidate.lower()
            if low in {
                "ти",
                "тебе",
                "імя",
                "ім'я",
                "ім’я",
                "бот",
                "хто",
                "що",
                "як",
                "де",
                "коли",
                "чому",
                "навіщо",
                "who",
                "what",
                "why",
                "how",
            }:
                continue
            return candidate

        return None

    def _record_fallback(
        self,
        user_text: str,
        session_id: str,
        queue_path: str = DEFAULT_FALLBACK_QUEUE_PATH,
    ) -> None:
        text = user_text.strip()
        if len(text) < 2:
            return

        queue = self._load_fallback_queue(queue_path)
        items = queue.get("items", [])
        now = datetime.now(timezone.utc).isoformat()

        found = False
        for item in items:
            if str(item.get("text", "")).strip().lower() == text.lower():
                item["text"] = text
                item["count"] = int(item.get("count", 0)) + 1
                item["last_seen"] = now
                item["last_session"] = session_id
                found = True
                break

        if not found:
            items.append(
                {
                    "text": text,
                    "count": 1,
                    "last_seen": now,
                    "last_session": session_id,
                }
            )

        queue["items"] = items
        self._save_fallback_queue(queue_path, queue)

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

    def _put_model_in_cache(self, model_path: str, model: NeuralIntentModel) -> None:
        path = Path(model_path)
        resolved = str(path.resolve())
        mtime = path.stat().st_mtime
        with self._lock:
            self._model_cache[resolved] = (mtime, model)

    def _get_model(self, model_path: str) -> NeuralIntentModel:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл моделі не знайдено: {model_path}")
        resolved = str(path.resolve())
        mtime = path.stat().st_mtime

        with self._lock:
            cached = self._model_cache.get(resolved)
            if cached is not None and cached[0] == mtime:
                return cached[1]

        model = NeuralIntentModel.load(path)
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

    @staticmethod
    def _load_fallback_queue(queue_path: str) -> dict:
        path = Path(queue_path)
        if not path.exists():
            return {"items": []}

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"items": []}

        if not isinstance(data, dict):
            return {"items": []}
        items = data.get("items", [])
        if not isinstance(items, list):
            return {"items": []}
        return {"items": items}

    @staticmethod
    def _save_fallback_queue(queue_path: str, data: dict) -> None:
        path = Path(queue_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")


def create_module() -> ChatModule:
    return ChatModule()
