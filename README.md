# NVIDIA Nemotron Model Reasoning Challenge

LoRA fine-tuning pipeline for the NVIDIA Nemotron-3-Nano-30B reasoning model on Kaggle.

## Overview

This project explores parameter-efficient fine-tuning for logical reasoning tasks using NVIDIA's Nemotron-3-Nano-30B model. The work was developed for the Kaggle NVIDIA Nemotron Model Reasoning Challenge, where the goal is to improve model reasoning performance using a LoRA adapter submission.

## Key Features

- Fine-tuning of Nemotron-3-Nano-30B using LoRA
- PEFT-based adapter training
- Mamba-SSM compatible setup
- RTX Pro 6000 GPU execution
- Kaggle-ready submission pipeline
- Adapter packaging into `submission.zip`

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Mamba-SSM
- Kaggle Notebooks
- NVIDIA RTX Pro 6000

## Workflow

1. Load the official Nemotron model from Kaggle Models.
2. Attach a LoRA adapter with rank 32.
3. Train the adapter on reasoning puzzle data.
4. Save the trained adapter.
5. Package the adapter as `submission.zip`.
6. Submit the adapter to Kaggle for evaluation.

## Dataset

The competition dataset contains logical reasoning puzzles with prompts and ground-truth answers. The tasks include structured reasoning patterns such as binary transformations, encryption rules, Roman numeral conversions, unit transformations, and algebraic reasoning.

## Current Experiment

- Model: Nemotron-3-Nano-30B-A3B-BF16
- Fine-tuning method: LoRA
- LoRA rank: 32
- Training steps: 200
- Accelerator: NVIDIA RTX Pro 6000
- Submission format: LoRA adapter ZIP

## Project Goal

The goal of this project is to build a strong reasoning-model fine-tuning workflow and iteratively improve leaderboard performance through adapter tuning, prompt formatting, and training optimization.

## Status

Initial LoRA adapter submission completed. Further experiments will focus on longer training, validation monitoring, and hyperparameter tuning.
