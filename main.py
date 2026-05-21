from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from core import Kernel, ModuleSpec


def build_kernel() -> Kernel:
    specs = [
        ModuleSpec(name="chat", entrypoint="modules.chat.entry"),
        ModuleSpec(name="snake", entrypoint="modules.snake.entry"),
    ]
    return Kernel(specs=specs)


def _metrics_record_base() -> dict[str, str]:
    from datetime import datetime, timezone

    return {"timestamp_utc": datetime.now(timezone.utc).isoformat()}


def _append_metrics_record(metrics_path: str, record: dict[str, Any]) -> Path:
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full_record = {**_metrics_record_base(), **record}
    suffix = path.suffix.lower()

    if suffix == ".json":
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = []
            if not isinstance(payload, list):
                payload = []
        else:
            payload = []
        payload.append(full_record)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    if suffix == ".jsonl":
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
        return path

    # Default: append as CSV.
    fieldnames = list(full_record.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(full_record)
    return path


def cmd_modules(kernel: Kernel) -> None:
    print("Статус модулів:")
    for status in kernel.list_module_status():
        print(f"  - {status.name} (loaded={status.loaded}, entrypoint={status.entrypoint})")


def cmd_train(kernel: Kernel, args: argparse.Namespace) -> None:
    module = kernel.load(args.module)
    stats = module.train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        model_path=args.model_path,
        seed=args.seed,
        log_every=args.log_every,
        resume=not args.fresh_start,
        goal_score=args.goal_score,
    )
    print(
        "[змійка-тренування] "
        f"episodes={stats.episodes}, "
        f"avg_score_last_100={stats.avg_score_last_100:.2f}, "
        f"best_score={stats.best_score}, "
        f"best_fill={stats.best_fill_percent:.1f}%, "
        f"wins={stats.wins}, "
        f"q_states={stats.q_states}, "
        f"total_episodes={stats.total_episodes}, "
        f"model={stats.model_path}, "
        f"goal={stats.goal_score_requested}->{stats.goal_score_effective}, "
        f"resumed={stats.resumed_from_model}, "
        f"epsilon={stats.epsilon_start:.4f}->{stats.epsilon_end:.4f}"
    )
    if args.metrics_path:
        out_path = _append_metrics_record(
            args.metrics_path,
            {
                "run_type": "train",
                "module": "snake",
                "episodes": stats.episodes,
                "avg_score_last_100": f"{stats.avg_score_last_100:.6f}",
                "best_score": stats.best_score,
                "best_fill_percent": f"{stats.best_fill_percent:.4f}",
                "wins": stats.wins,
                "q_states": stats.q_states,
                "total_episodes": stats.total_episodes,
                "goal_score_requested": stats.goal_score_requested,
                "goal_score_effective": stats.goal_score_effective,
                "epsilon_start": f"{stats.epsilon_start:.8f}",
                "epsilon_end": f"{stats.epsilon_end:.8f}",
                "resumed_from_model": stats.resumed_from_model,
                "model_path": stats.model_path,
            },
        )
        print(f"[metrics] saved: {out_path}")


def cmd_play(kernel: Kernel, args: argparse.Namespace) -> None:
    module = kernel.load(args.module)
    stats = module.play(
        episodes=args.episodes,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        model_path=args.model_path,
        seed=args.seed,
        goal_score=args.goal_score,
    )
    print(
        "[змійка-оцінка] "
        f"episodes={stats.episodes}, "
        f"avg_score={stats.avg_score:.2f}, "
        f"best_score={stats.best_score}, "
        f"best_fill={stats.best_fill_percent:.1f}%, "
        f"wins={stats.wins}, "
        f"goal={stats.goal_score_requested}->{stats.goal_score_effective}, "
        f"model={stats.model_path}"
    )
    if args.metrics_path:
        model_info = module.get_model_info(model_path=args.model_path)
        out_path = _append_metrics_record(
            args.metrics_path,
            {
                "run_type": "eval",
                "module": "snake",
                "episodes": stats.episodes,
                "avg_score": f"{stats.avg_score:.6f}",
                "best_score": stats.best_score,
                "best_fill_percent": f"{stats.best_fill_percent:.4f}",
                "wins": stats.wins,
                "q_states": model_info.q_states,
                "total_episodes": model_info.total_episodes,
                "goal_score_requested": stats.goal_score_requested,
                "goal_score_effective": stats.goal_score_effective,
                "model_path": stats.model_path,
            },
        )
        print(f"[metrics] saved: {out_path}")


def cmd_train_chat(kernel: Kernel, args: argparse.Namespace) -> None:
    module = kernel.load("chat")
    stats = module.train(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        seed=args.seed,
    )
    print(
        "[чат-тренування] "
        f"intents={stats.intents}, "
        f"examples={stats.examples}, "
        f"vocab={stats.vocab_size}, "
        f"model={stats.model_path}"
    )


def cmd_chat(kernel: Kernel, args: argparse.Namespace) -> None:
    module = kernel.load("chat")

    model_path = Path(args.model_path)
    if args.auto_train and not model_path.exists():
        module.train(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            seed=args.seed,
        )
        print(f"[інфо] Модель не знайдена. Автотренування виконано: {args.model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Файл моделі не знайдено: {args.model_path}. "
            "Запусти: python main.py train-chat"
        )

    print("Чат готовий до роботи. Команди:")
    print("  /exit")
    print("  /teach intent | example | response")

    while True:
        try:
            user_text = input("ти> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[інфо] Чат завершено.")
            break

        if not user_text:
            continue
        if user_text == "/exit":
            print("[інфо] Чат завершено.")
            break

        if user_text.startswith("/teach "):
            payload = user_text[len("/teach ") :].strip()
            parts = [p.strip() for p in payload.split("|", 2)]
            if len(parts) != 3:
                print("бот> Формат: /teach intent | example | response")
                continue

            intent, example, response = parts
            module.teach(
                intent=intent,
                example=example,
                response=response,
                dataset_path=args.dataset_path,
            )
            module.train(
                dataset_path=args.dataset_path,
                model_path=args.model_path,
                seed=args.seed,
            )
            print(f"бот> Додано інтент '{intent}'. Модель перенавчено.")
            continue

        reply = module.reply(
            user_text=user_text,
            model_path=args.model_path,
            confidence_threshold=args.threshold,
        )
        print(f"бот> {reply.text} (intent={reply.intent}, conf={reply.confidence:.2f})")


def cmd_test_chat(kernel: Kernel, args: argparse.Namespace) -> None:
    module = kernel.load("chat")
    stats = module.train(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        seed=args.seed,
    )
    print(
        "[чат-тест] "
        f"model trained: intents={stats.intents}, examples={stats.examples}, vocab={stats.vocab_size}"
    )

    cases = [
        ("привіт", "greeting"),
        ("привітос", "greeting"),
        ("що ти можеш", "capabilities"),
        ("чому так", "why"),
        ("дякую", "thanks"),
        ("спс", "thanks"),
        ("давай тебе буде звати Timo", "set_name"),
        ("хто ти", "name"),
        ("ти хто", "name"),
        ("но ми тебе ж називали Тімо", "set_name"),
        ("хто ти", "name"),
        ("тепер тебе звати Тімо", "set_name"),
        ("ні ти Timo", "set_name"),
        ("як навчити тебе краще", "train_help"),
        ("хто тебе створив", "creator"),
        ("до побачення", "bye"),
        ("до звязку", "bye"),
        ("на все добре", "bye"),
        ("який курс біткоїна зараз", "fallback"),
    ]

    session_id = "benchmark"
    passed = 0
    for idx, (question, expected_intent) in enumerate(cases, start=1):
        reply = module.reply(
            user_text=question,
            model_path=args.model_path,
            confidence_threshold=args.threshold,
            session_id=session_id,
        )
        ok = reply.intent == expected_intent
        if ok:
            passed += 1
        status = "OK" if ok else "MISS"
        print(
            f"[{idx:02d}] {status} "
            f"q='{question}' -> intent={reply.intent} (exp={expected_intent}), "
            f"conf={reply.confidence:.2f}, text='{reply.text}'"
        )

    total = len(cases)
    print(f"[чат-тест] accuracy={passed}/{total} ({(100.0 * passed / total):.1f}%)")


def cmd_app() -> None:
    from app import run_app

    run_app()


def cmd_app_chat() -> None:
    from apps.chat_app import run_chat_app

    run_chat_app()


def cmd_app_snake() -> None:
    from apps.snake_app import run_snake_app

    run_snake_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local AI ядро з окремими застосунками."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("modules", help="Показати статус модулів.")

    train = subparsers.add_parser("train", help="Тренувати модель.")
    train.add_argument("module", choices=["snake"])
    train.add_argument("--episodes", type=int, default=3000)
    train.add_argument("--max-steps", type=int, default=250, help="0 = м'яко без обмеження довжини епізоду.")
    train.add_argument("--width", type=int, default=10)
    train.add_argument("--height", type=int, default=10)
    train.add_argument("--model-path", type=str, default="models/snake_q.pkl")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log-every", type=int, default=250)
    train.add_argument("--fresh-start", action="store_true", help="Почати з нуля (ігнорувати попередню модель).")
    train.add_argument("--goal-score", type=int, default=100, help="Бажаний рахунок перемоги (обрізається до меж сітки).")
    train.add_argument(
        "--metrics-path",
        type=str,
        default="",
        help="Опційно: шлях для експорту метрик (CSV/JSON/JSONL).",
    )

    play = subparsers.add_parser("play", help="Оцінити натреновану модель.")
    play.add_argument("module", choices=["snake"])
    play.add_argument("--episodes", type=int, default=10)
    play.add_argument("--max-steps", type=int, default=250, help="0 = м'яко без обмеження довжини епізоду.")
    play.add_argument("--width", type=int, default=10)
    play.add_argument("--height", type=int, default=10)
    play.add_argument("--model-path", type=str, default="models/snake_q.pkl")
    play.add_argument("--seed", type=int, default=123)
    play.add_argument("--goal-score", type=int, default=100, help="Бажаний рахунок перемоги (обрізається до меж сітки).")
    play.add_argument(
        "--metrics-path",
        type=str,
        default="",
        help="Опційно: шлях для експорту метрик (CSV/JSON/JSONL).",
    )

    train_chat = subparsers.add_parser("train-chat", help="Тренувати чат-модель.")
    train_chat.add_argument("--dataset-path", type=str, default="modules/chat/data/intents.json")
    train_chat.add_argument("--model-path", type=str, default="models/chat_intent.pkl")
    train_chat.add_argument("--seed", type=int, default=42)

    chat = subparsers.add_parser("chat", help="Запустити консольний чат.")
    chat.add_argument("--dataset-path", type=str, default="modules/chat/data/intents.json")
    chat.add_argument("--model-path", type=str, default="models/chat_intent.pkl")
    chat.add_argument("--threshold", type=float, default=0.25)
    chat.add_argument("--seed", type=int, default=42)
    chat.add_argument("--auto-train", action="store_true")

    test_chat = subparsers.add_parser("test-chat", help="Прогнати швидкий тест чату.")
    test_chat.add_argument("--dataset-path", type=str, default="modules/chat/data/intents.json")
    test_chat.add_argument("--model-path", type=str, default="models/chat_intent.pkl")
    test_chat.add_argument("--threshold", type=float, default=0.25)
    test_chat.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("app", help="Запустити GUI-лаунчер.")
    subparsers.add_parser("app-chat", help="Запустити окремий застосунок чату.")
    subparsers.add_parser("app-snake", help="Запустити окремий застосунок змійки.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = build_kernel()

    if args.command == "modules":
        cmd_modules(kernel)
    elif args.command == "train":
        cmd_train(kernel, args)
    elif args.command == "play":
        cmd_play(kernel, args)
    elif args.command == "train-chat":
        cmd_train_chat(kernel, args)
    elif args.command == "chat":
        cmd_chat(kernel, args)
    elif args.command == "test-chat":
        cmd_test_chat(kernel, args)
    elif args.command == "app":
        cmd_app()
    elif args.command == "app-chat":
        cmd_app_chat()
    elif args.command == "app-snake":
        cmd_app_snake()
    else:
        raise ValueError(f"Невідома команда: {args.command}")


if __name__ == "__main__":
    main()
