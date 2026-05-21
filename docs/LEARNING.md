# Learning Logic

## What “Learning” Means In This Project
Learning here means updating local model parameters from local training data or training episodes, then saving those parameters to disk for reuse.

This repository has two different learning systems:
- Chat intent learning (supervised classification)
- Snake agent learning (reinforcement learning)

It is not large language model training.

## Chat Intent Learning

### Data Source
Chat training reads labeled examples from:
- `modules/chat/data/intents.json`

Each intent contains:
- `name`
- `examples`
- `responses`

### Model Type
`modules/chat/model.py` implements `NeuralIntentModel`:
- Small MLP classifier
- Bag-of-features style input
- Feature extraction from:
  - word tokens
  - token bigrams
  - character trigrams

The model predicts:
- intent label
- confidence score
- extra confidence context (margin, token coverage)

### How Training Works
1. Dataset is parsed.
2. Features are built from examples.
3. Vocabulary and intent indices are created.
4. MLP weights are initialized.
5. SGD updates weights for configured epochs.
6. Trained model is saved to `models/chat_intent.pkl` (default).

### How `/teach` Helps
In chat CLI (`python main.py chat`):
- You can use:
  - `/teach intent | example | response`

What happens:
1. Example/response are appended to the matching intent (or a new intent is created).
2. Dataset JSON is saved.
3. Model retraining is triggered immediately.

This is supervised incremental improvement through new labeled examples.

### Fallback Queue Learning
When the model is uncertain, chat can output fallback and record unknown prompts in:
- `modules/chat/data/fallback_queue.json`

In the GUI developer mode, these fallback prompts can be converted into labeled examples and retrained.

## Snake Reinforcement Learning

## Algorithm Used In Code
Important accuracy note:
- Class name is `QLearningSnakeAgent` for compatibility.
- Real implementation is DQN-style (not tabular Q-learning).

DQN components in code:
- Experience replay buffer
- Online network + target network
- Epsilon-greedy exploration
- Mini-batch gradient updates
- Target sync interval

## Environment And Episodes
`SnakeEnv` defines:
- state representation (discrete feature vector)
- step transitions
- terminal conditions (`wall`, `self`, `starvation`, `goal`)
- rewards and penalties

An episode starts with `reset()` and ends on terminal condition or step limit.

## Rewards And Penalties
Current reward logic in `modules/snake/env.py`:
- `+15` for eating food
- `+25` additional bonus for reaching goal score
- `-10` for collision or starvation terminal states
- per-step shaping:
  - base `-0.05`
  - `+0.20` if moved closer to food
  - `-0.10` if moved farther/equal
- loop penalty (default `0.20`) for repeating recent head positions

These rewards drive agent behavior and learning speed.

## Epsilon (Exploration)
Epsilon controls random actions during training:
- high epsilon -> more exploration
- low epsilon -> more exploitation

The agent decays epsilon after each episode:
- starts near `1.0` by default
- decays toward `epsilon_min`

Training output reports:
- `epsilon_start`
- `epsilon_end`

## Q-States / Model States
Training output currently reports `q_states`, but for this DQN implementation it represents model parameter count (not tabular state count).

Use it as a model-size indicator rather than a count of discovered board states.

## Model Resume
Snake training can resume from existing model:
- default behavior: resume if model file exists
- CLI override: `--fresh-start` to ignore prior model

When resumed:
- network weights are loaded
- epsilon and episode counters continue
- additional training builds on previous learning

## Interpreting Training Results
Useful signals:
- `avg_score_last_100` trending up
- `best_score` increasing over time
- `wins` becoming more frequent
- `epsilon` decaying while performance holds or improves

Weak signals:
- flat or declining average score over long runs
- high collision frequency without improvement
- no progress between fresh and resumed runs

For practical evaluation workflow, see:
- `docs/EVALUATION.md`
