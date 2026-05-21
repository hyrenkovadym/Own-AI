from __future__ import annotations

import json
from pathlib import Path

from modules.chat.entry import ChatModule


def _write_dataset(path: Path) -> None:
    payload = {
        "intents": [
            {
                "name": "greeting",
                "examples": ["hello", "hi"],
                "responses": ["Hello there!"],
            },
            {
                "name": "bye",
                "examples": ["bye"],
                "responses": ["See you!"],
            },
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_chat_train_and_reply(tmp_path: Path) -> None:
    dataset_path = tmp_path / "intents.json"
    model_path = tmp_path / "chat_intent.pkl"
    _write_dataset(dataset_path)

    module = ChatModule()
    stats = module.train(dataset_path=str(dataset_path), model_path=str(model_path), seed=7)

    assert stats.intents == 2
    assert stats.examples == 3
    assert stats.vocab_size > 0
    assert model_path.exists()

    reply = module.reply("hello", model_path=str(model_path), confidence_threshold=0.2, session_id="t1")
    assert reply.intent == "greeting"
    assert reply.confidence >= 0.9


def test_chat_teach_updates_dataset_and_model(tmp_path: Path) -> None:
    dataset_path = tmp_path / "intents.json"
    model_path = tmp_path / "chat_intent.pkl"
    _write_dataset(dataset_path)

    module = ChatModule()
    module.train(dataset_path=str(dataset_path), model_path=str(model_path), seed=3)

    module.teach(
        intent="thanks",
        example="thanks",
        response="You are welcome.",
        dataset_path=str(dataset_path),
    )
    module.train(dataset_path=str(dataset_path), model_path=str(model_path), seed=3)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    intent_names = {item["name"] for item in data["intents"]}
    assert "thanks" in intent_names

    reply = module.reply("thanks", model_path=str(model_path), confidence_threshold=0.2, session_id="t2")
    assert reply.intent == "thanks"
