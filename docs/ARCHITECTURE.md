# Architecture

## Project Context
Repository name: `Own-AI`  
Professional title: **Local Modular AI Learning System**

This project is a local modular AI/ML runtime. It is not an LLM system and does not rely on external AI APIs.

## High-Level Architecture

```text
User (CLI or GUI)
    |
    v
main.py / app.py
    |
    v
Kernel (core/kernel.py)
    |-- loads "chat" module -> modules/chat/entry.py -> ChatModule
    |-- loads "snake" module -> modules/snake/entry.py -> SnakeModule
    |
    v
Local model files (models/*.pkl)
```

## Core Kernel Responsibilities
`core/kernel.py` provides the runtime orchestration layer:
- Registers module metadata (`ModuleSpec`)
- Lazy-loads modules by name
- Creates module instances via entrypoint `create_module()`
- Tracks loaded instances
- Supports lifecycle operations:
  - `load`
  - `unload`
  - `reload`
  - `unload_all`
- Calls optional hooks:
  - `on_load(kernel)`
  - `on_unload()`
- Exposes module status (`list_module_status`)
- Uses thread lock (`RLock`) for safe concurrent access

## Module Loading
The project configures modules in both CLI and GUI helpers with:
- `ModuleSpec(name="chat", entrypoint="modules.chat.entry")`
- `ModuleSpec(name="snake", entrypoint="modules.snake.entry")`

When loaded:
1. `Kernel.load("chat")` or `Kernel.load("snake")`
2. Python imports module entrypoint
3. Kernel calls `create_module()`
4. Optional `on_load` hook receives kernel reference

## Module Lifecycle
- First load creates a singleton instance per module name.
- Repeated loads return the cached instance.
- `reload(name)` unloads and creates a fresh instance.
- `unload(name)` removes instance and runs optional `on_unload`.

This keeps modules isolated and independently replaceable.

## Chat Module Responsibilities
`modules/chat/entry.py`:
- Train local intent model from dataset JSON
- Predict intent + confidence for user text
- Return mapped response templates
- Support incremental supervised teaching (`teach`)
- Maintain lightweight per-session memory and custom name memory
- Maintain fallback queue for low-confidence/unknown prompts
- Provide local programmer helper mode (`LocalProgrammerAgent`)

Model implementation:
- `modules/chat/model.py`
- Local MLP intent classifier with handcrafted text features
- No external API calls

## Snake Module Responsibilities
`modules/snake/entry.py`:
- Train agent with configurable episodes/steps/grid size
- Evaluate trained agent
- Produce rollout frames for GUI visualization
- Provide live-training frames for real-time UI rendering
- Save/load model for resume across runs
- Report training/evaluation stats

Environment:
- `modules/snake/env.py` (grid world, rewards, terminal conditions)

Agent:
- `modules/snake/agent.py`
- Class name is `QLearningSnakeAgent` for backward compatibility
- Implementation is DQN-style (online/target networks + replay)

## CLI Flow
Entry file: `main.py`

Typical flow:
1. Parse command args
2. Build kernel
3. Load requested module
4. Execute command logic
5. Print stats to stdout
6. Optionally export Snake metrics (`--metrics-path`)

Main command groups:
- `modules`
- `train snake`
- `play snake`
- `train-chat`
- `chat`
- `test-chat`
- `app`, `app-chat`, `app-snake`

## GUI Flow
Files:
- `app.py` (launcher)
- `apps/chat_app.py`
- `apps/snake_app.py`

Common flow:
1. GUI app builds kernel
2. User action triggers module method call
3. Work runs on background thread
4. UI queue posts updates to main thread
5. Logs/frames/stats are rendered in widgets

## Model Storage
Default paths:
- Chat: `models/chat_intent.pkl`
- Snake: `models/snake_q.pkl`

Storage behavior:
- Chat training overwrites model file with current trained parameters
- Snake training saves DQN parameters + training metadata
- Snake can resume from existing model unless fresh start is requested

## Current Limitations
- Chat is classification-based, not generative language modeling.
- Intent quality depends on dataset diversity and labeling quality.
- Snake performance depends on reward tuning and training budget.
- No experiment database; metrics export is file-based (CSV/JSON/JSONL).
- GUI has no automated tests in CI.
