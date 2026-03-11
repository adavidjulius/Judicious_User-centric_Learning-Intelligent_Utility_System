# JULIUS
Judicious User-centric Learning Intelligent Utility System

JULIUS is an experimental AI system focused on building a **scientifically capable assistant** specialized in:

- Mathematics
- Physics
- Chemistry
- Scientific reasoning

The project explores building a **custom STEM-focused language model** with integrated voice interface and modular research infrastructure.

---

## Core Goals

1. Train a STEM-specialized LLM
2. Build a dataset pipeline for scientific material
3. Integrate speech interfaces
4. Provide CLI + web interface
5. Maintain an open research workflow

---

## Architecture

JULIUS is divided into modular systems:

| Module | Purpose |
|------|------|
| `julius_core` | Model loading and inference |
| `julius_training` | Training pipelines and LoRA fine-tuning |
| `julius_data` | Dataset preparation and storage |
| `julius_voice` | Speech interface (TTS + STT) |
| `julius_interface` | CLI, API, and UI layers |
| `julius_experiments` | Research experiments and benchmarks |

---

## Current Model Base

Initial experiments use:

Mistral-7B-v0.1 + QLoRA fine-tuning

Training is performed using:

- Kaggle T4 GPUs
- Colab
- HuggingFace ecosystem

---

## Project Status

Early research stage.

Focus areas:

- dataset curation
- fine-tuning experiments
- inference optimization

---

## Roadmap

Phase 1  
STEM dataset preparation

Phase 2  
Domain fine-tuning

Phase 3  
Voice interaction system

Phase 4  
Full JULIUS assistant

---

## License

MIT License
