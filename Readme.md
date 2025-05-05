# Final Project

## Overview

This project implements a comprehensive framework for goal-oriented robot manipulation learning based on the GR-MG (Goal Recognition - Motion Generation) architecture. The system leverages visual goal generation and policy learning to enable robots to understand and execute manipulation tasks from natural language instructions and visual observations.

## Architecture

**Goal Generation Module:** Based on InstructPix2Pix, this module transforms the current state and task instructions into a goal state image.

## Preparation

- Linux operating system
- NVIDIA GPU with CUDA 12.1 support
- Docker and nvidia-docker
- 24GB+ GPU memory recommended
- Python 3.10
- Git

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/FinalProject.git
cd FinalProject
```

2. Synchronize Git submodules

```bash
make sync
```

3. Build the Docker image

```bash
make build
```

4. Download required models

```bash
# Download InstructPix2Pix model
make download CMD=1
```

### Data Preparation

1. Use RoboTwin to generate PKL files

Follow the instruction in INSTALLATION.md and README.md in **/lib/RoboTwin**.

2. Extract RGB images from PKL files

```bash
make output-img
```

3. Format the dataset

```bash
make output-dataset
```

4. Generate metadata files

```bash
make output-meta
```

## Training Pipeline

### Goal Generation Training

Train the goal generation model to predict goal states from current observations:

```bash
make run CMD=1
```

## Evaluation Pipeline

### Goal Generation Evaluation

Evaluate the goal generation model:

```bash
make run CMD=2
```

## Project Structure

```
FinalProject/
├── GR-MG/                    # Main implementation
│   ├── goal_gen/             # Goal generation module
├── config/                   # Configuration files
│   └── GR-MG/
│       ├── goal_gen/
│   lib/
│   └── RoboTwin /            # RoboTwin generate datasets
├── resources/                # Resources and assets
├── script/                   # Utility scripts
│   ├── extract_pkl_img.py    # Extract images from PKL files
│   ├── format_data.py        # Format dataset
│   ├── output_meta.py        # Generate metadata
│   ├── hfd_config.json       # HuggingFace username and token
│   └── hfd.sh                # HuggingFace download script
├── Dockerfile                # Docker configuration
├── Makefile                  # Build and run commands
└── wandb_key.txt             # Weights & Biases API key
```

## Experiment Tracking

This project uses Weights & Biases for experiment tracking. To use it:

1. Create a wandb_key.txt file with your API key
2. Training runs will automatically log metrics and visualizations to your W&B account

## Customization

1. Modify configuration files in the config directory to adjust hyperparameters
2. Training datasets are expected in data
3. Results are saved to results

## Acknowledgments

1. [InstrcutPix2Pix](https://hf-mirror.com/timbrooks/instruct-pix2pix)
2. [RoboTwin](http://github.com/TianxingChen/RoboTwin) or the dataset
3. [GR-MG](https://github.com/bytedance/GR-MG)
