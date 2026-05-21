# Local Modular AI Learning System
### Trainable Chat Intent Model + Snake Reinforcement Learning Agent

Repository name: `Own-AI`  
Professional project title for portfolio presentation: **Local Modular AI Learning System**

> This is **not** a ChatGPT clone or an LLM.  
> It is a local AI/ML learning system for modular runtime design, intent classification, and reinforcement learning experiments.

## Overview
This project demonstrates how to build a modular local AI runtime in Python with independently loadable modules, local model training, model persistence, CLI controls, and desktop GUI demos.

It includes:
- A runtime kernel that loads/unloads modules dynamically
- A trainable chat intent module (local neural classifier)
- A Snake reinforcement learning module (local DQN-style agent)
- CLI workflows for training/evaluation
- Desktop apps for chat and Snake visualization

## What This Project Demonstrates
- Python architecture and modular design
- Local AI/ML workflows without external AI APIs
- Intent classification training and incremental teaching
- Reinforcement learning fundamentals (state, reward, exploration, replay)
- Model save/load and training resume
- CLI and GUI integration
- Practical testing and CI setup

## Main Modules
- `core/`
  - Runtime kernel (`Kernel`) with lazy loading, lifecycle hooks, and thread-safe module handling.
- `modules/chat/`
  - Local trainable chat intent model and chat runtime logic.
- `modules/snake/`
  - Snake environment and reinforcement learning agent (DQN implementation, class name kept for backward compatibility).
- `main.py`
  - CLI entrypoint for module status, training, evaluation, and app launching.
- `apps/chat_app.py`, `apps/snake_app.py`, `app.py`
  - Desktop GUI apps and launcher.

## Features
- Modular kernel (`load`, `unload`, `reload`, `list_module_status`)
- Chat intent training from dataset JSON
- `/teach` command in CLI for incremental dataset growth and retraining
- Optional fallback queue for unknown chat prompts
- Snake batch training, live training, and evaluation
- Training resume from saved model
- Optional training/evaluation metrics export to CSV/JSON/JSONL
- Local-only execution (no external AI API dependency)

## Tech Stack
- Python 3.10+ (tested with Python 3.12)
- NumPy (for RL model computations)
- Tkinter (desktop GUI)
- Pytest (tests)
- GitHub Actions (CI)

## Architecture Overview
High-level flow:

1. CLI or GUI requests a module.
2. `Kernel` lazily loads module entrypoint (`create_module()`).
3. Module handles training/reply/evaluation logic.
4. Models are persisted to `models/` and reused on next runs.

Detailed architecture docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## How The Module System Works
- Module specifications are registered via `ModuleSpec(name, entrypoint)`.
- `Kernel.load(name)` imports entrypoint and calls `create_module()`.
- Optional lifecycle hooks:
  - `on_load(kernel)`
  - `on_unload()`
- The kernel tracks loaded state and supports `reload`/`unload_all`.

## How The Chat Module Learns
- Training data comes from `modules/chat/data/intents.json`.
- `NeuralIntentModel` builds token/feature vectors and trains a small local MLP classifier.
- Responses are chosen from intent response templates.
- `/teach intent | example | response` adds new examples/responses and retrains.
- Unknown/low-confidence prompts can be queued to `fallback_queue.json` for later supervised teaching.

Detailed learning notes: [`docs/LEARNING.md`](docs/LEARNING.md)

## How The Snake RL Module Learns
- Environment: `SnakeEnv` (headless grid environment).
- Agent: `QLearningSnakeAgent` class name (legacy naming), implementation is DQN-style with:
  - replay buffer
  - online + target network
  - epsilon-greedy exploration
  - gradient updates with NumPy
- Training supports resume from existing model and epsilon decay across episodes.

Evaluation guidance: [`docs/EVALUATION.md`](docs/EVALUATION.md)

## Model Persistence
Default model paths:
- Chat model: `models/chat_intent.pkl`
- Snake model: `models/snake_q.pkl`

Persisted artifacts include learned weights and metadata (for example epsilon and episode counters for Snake).  
Snake training can resume from an existing model unless `--fresh-start` is used.

## CLI Commands

### General
```bash
python main.py modules
python main.py app
python main.py app-chat
python main.py app-snake
```

### Chat
```bash
python main.py train-chat
python main.py chat --auto-train
python main.py test-chat
```

Teach interactively in chat CLI:
```text
/teach intent | example | response
```

### Snake Training
```bash
python main.py train snake --episodes 3000 --max-steps 250
```

Start from scratch (ignore saved model):
```bash
python main.py train snake --episodes 3000 --fresh-start
```

Export metrics (CSV/JSON/JSONL):
```bash
python main.py train snake --episodes 500 --metrics-path runs/snake_training_metrics.csv
```

### Snake Evaluation
```bash
python main.py play snake --episodes 20
```

Export evaluation metrics:
```bash
python main.py play snake --episodes 20 --metrics-path runs/snake_eval_metrics.json
```

## GUI Apps
- `python main.py app-chat`
  - Chat conversation, local training, optional developer controls, and fallback queue teaching.
- `python main.py app-snake`
  - Batch training, live training visualization, evaluation, rollout animation, model reset, and resume toggle.
- `python main.py app`
  - Lightweight launcher for chat/snake apps.

## Local Setup
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run tests:
```bash
pytest -q
```

## How To Train Chat
1. Ensure dataset exists at `modules/chat/data/intents.json`.
2. Run:
   ```bash
   python main.py train-chat
   ```
3. Start CLI chat:
   ```bash
   python main.py chat --auto-train
   ```
4. Teach new intent examples with:
   ```text
   /teach intent | example | response
   ```

## How To Train Snake
Quick run:
```bash
python main.py train snake --episodes 100 --max-steps 250
```

Longer run:
```bash
python main.py train snake --episodes 3000 --max-steps 250 --log-every 250
```

Resume behavior:
- default is resume if model exists
- use `--fresh-start` to ignore previous model

## How To Evaluate Snake
```bash
python main.py play snake --episodes 50 --max-steps 250
```

Compare fresh vs resumed training by running evaluation after:
- a fresh-start training run
- a resumed training run from existing model

## Screenshots / GIF Demos
Place media files in `docs/assets/` and update links below.

- Chat App Screenshot Placeholder  
  `docs/assets/chat_app_placeholder.png`
- Snake Live Training GIF Placeholder  
  `docs/assets/snake_live_training_placeholder.gif`
- Snake Evaluation Screenshot Placeholder  
  `docs/assets/snake_eval_placeholder.png`

## Example Training Output
Illustrative CLI output format (example values, not guaranteed results):

```text
[змійка-тренування] episodes=500, avg_score_last_100=8.42, best_score=17, best_fill=18.9%, wins=2, q_states=35043, total_episodes=12500, model=models/snake_q.pkl, goal=100->97, resumed=True, epsilon=0.1043->0.0681
[metrics] saved: runs/snake_training_metrics.csv
```

```text
[змійка-оцінка] episodes=20, avg_score=9.15, best_score=14, best_fill=15.6%, wins=0, goal=100->97, model=models/snake_q.pkl
[metrics] saved: runs/snake_eval_metrics.json
```

## Current Limitations
- Chat module is intent classification, not generative conversation.
- Dataset quality and coverage directly limit chat behavior.
- Snake training quality depends on reward shaping and episode budget.
- CLI/UI messages are partly Ukrainian in the current implementation.
- No distributed training or large-scale experiment tracking.

## Future Improvements
- Richer metrics dashboards and learning curves
- Hyperparameter presets and config files
- Better chat dataset tooling and validation
- Additional local AI modules through plugin/module interface
- Packaging and release workflow improvements

See full roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)

## What This Project Demonstrates To Employers
This repository showcases practical local AI engineering skills:
- designing a modular Python runtime
- implementing trainable local ML components
- integrating RL environment + agent training loops
- handling model persistence and resume workflows
- building both CLI and GUI interfaces
- adding tests, CI, and technical documentation

Portfolio message:

> “I built a local modular AI learning system in Python with a trainable chat intent module and a reinforcement learning Snake agent. The project demonstrates modular architecture, local model training, model persistence, CLI control, GUI visualization, testing, and technical documentation.”
