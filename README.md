# Local AI Core

Local modular AI project without GPT API.

## Architecture

- `core/` - runtime kernel (lazy loading, thread-safe module lifecycle, reload/unload).
- `modules/chat/` - trainable local intent chat module.
- `modules/snake/` - Snake RL module (DQN self-learning).
- `apps/chat_app.py` - standalone desktop chat app.
- `apps/snake_app.py` - standalone desktop snake app with visualization.
- `app.py` - lightweight launcher for choosing app.
- `main.py` - CLI entrypoint.

## Run

```bash
python main.py modules
python main.py app
python main.py app-chat
python main.py app-snake
```

For live snake learning visualization:

1. Run `python main.py app-snake`
2. Click `Train` for fast non-visual batch learning (`Scenes` episodes)
3. Click `Learn Live` to watch learning with animation
4. Click `Auto Learn` for continuous self-learning until `Stop`
5. Watch logs with fail reasons (`wall`, `self`, `starvation`) and metrics
6. Optional: set `Hint Action` (`straight/right/left`) or type `Text Hint` (for example `line by line`) and click apply to guide next steps
7. Set `Hint Steps=auto` to apply the hint for exactly one episode

Use `Resume Model` in Snake app to keep learning memory between runs.
For visible improvement, prefer 100+ episodes (25 is mostly a quick demo).
For stronger progress on larger boards, use `Auto Learn` for long runs.
Use `Reset Memory` in Snake app to delete saved model with confirmation.
Snake app shows cumulative `Total Episodes` from saved model memory.
Model storage path is locked in UI (not editable).
Use `Win Score` to define victory target (auto-clamped to board max possible score).
`Train` and `Auto Learn` can be stopped safely with `Stop`.

## Chat CLI

```bash
python main.py train-chat
python main.py chat --auto-train
python main.py test-chat
```

Teach from chat CLI:

```text
/teach intent | example | response
```

## Chat App (simple + programmer mode)

1. Run `python main.py app-chat`
2. By default use simple chat view (no extra panels).
3. Enable `Режим розробника` if you want dataset/teaching controls.
4. Use `Role=programmer` to switch to coding assistant behavior.

Programmer mode is fully local and has no external API dependency.
It supports practical commands:

- `/help`
- `/ls [path]`
- `/read <file>`
- `/find <text>`
- `/tests`
- `/plan <task>`

Fallback self-learning queue:

- Unknown/fallback user messages are stored in `modules/chat/data/fallback_queue.json`.
- In developer mode click `Взяти Fallback` and `Навчити з Fallback` to quickly convert unknown phrases into new training examples.

## Snake CLI

```bash
python main.py train snake --episodes 3000
python main.py play snake --episodes 20
```

## What changed

- Separate desktop apps for chat and snake to avoid one heavy combined UI.
- Snake app now has live Canvas visualization of model gameplay.
- Kernel upgraded with:
  - thread-safe load/unload/reload
  - module lifecycle hooks (`on_load`, `on_unload`)
  - module status introspection
- Chat module upgraded with:
  - in-memory model cache (faster replies)
  - session memory fallback for low-confidence inputs
  - explicit session reset
# Own-AI
