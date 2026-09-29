# Co-Evolutionary Knowledge-Guided Reinforcement Learning for Crane Scheduling in Road-Rail Yard

Official implementation of:

**Co-Evolutionary Knowledge-Guided Reinforcement Learning for Crane Scheduling in Road-Rail Yard**

This repository provides the implementation of a dynamic knowledge-guided reinforcement learning framework for automated yard crane (AYC) scheduling in road-rail intermodal yards.

The proposed method integrates:
- Soft Actor-Critic (SAC) reinforcement learning;
- Hierarchical decision-making;
- Engineering-based scheduling knowledge;
- Policy-driven rule evolution;
- Knowledge-policy co-evolution mechanism.

The objective is to achieve efficient and robust crane scheduling under complex operational disturbances.

---

## Overview

Road-rail intermodal yards involve strong interactions among trains, trucks, containers, storage locations, transfer points, and automated yard cranes. Dynamic disturbances such as train delays, equipment failures, task insertion, task cancellation, and urgency changes make real-time scheduling challenging.

This project formulates the yard operation as a Markov Decision Process and proposes a knowledge-guided reinforcement learning framework.

The framework contains three main components:

1. **Yard Scheduling Environment**

A simulation environment models:
- Trains
- Trucks
- Containers
- Yard slots
- Transfer points
- Automated Yard Cranes (AYCs)

The scheduling objective minimizes weighted train completion time and truck waiting time.

2. **Hierarchical Decision-Making Mechanism**

The SAC policy outputs continuous preference vectors.

These preferences are decoded into executable scheduling decisions:

```
Task selection
      ↓
Destination selection
      ↓
Transfer-point selection
      ↓
AYC assignment
```

Engineering constraints and dynamic scheduling rules are incorporated to guarantee feasible decisions.

3. **Policy-Knowledge Co-Evolution**

The framework extracts scheduling rules from high-quality policy trajectories.

The generated rules are:

```
Policy trajectory
        ↓
Candidate rule generation
        ↓
Rule validation
        ↓
Rule feedback
        ↓
Policy improvement
```

This creates a closed-loop interaction between reinforcement learning and scheduling knowledge evolution.

---

# Repository Structure

```
.
├── config/
│   └── constants.py
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── experiments/
│   ├── run_v14_coevo_rgcd.py
│   └── run_rule_guided_evaluate.py
│
├── scripts/
│   ├── generate_instances.py
│   ├── generate_validation_instances.py
│   └── generate_template_rules.py
│
└── src/
    ├── core/
    │   ├── simulators.py
    │   ├── data_classes.py
    │   └── instance_parser.py
    │
    ├── env/
    │   ├── kg_env.py
    │   ├── disturbance.py
    │   ├── reward.py
    │   └── scoring.py
    │
    ├── eval/
    │   └── metrics.py
    │
    └── innovation_A/
        ├── evolution_manager.py
        ├── rule_guidance.py
        ├── rule_bonus.py
        ├── template_rules.py
        └── trajectory_collector.py
```

---

# Requirements

The implementation is tested with:

- Python 3.8
- PyTorch 2.4

Hardware used in experiments:

- Intel Core i7 CPU
- NVIDIA RTX 3050 Ti GPU
- 32 GB RAM

---

# Installation

Clone this repository:

```bash
git clone https://github.com/CZhu0066/Co-Evolutionary_Knowledge-Guided_RL.git

cd Co-Evolutionary_Knowledge-Guided_RL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

The repository contains generated scheduling instances:

```
data/

├── train/
├── val/
└── test/
```

Three main scenarios are provided:

- Two loading trains
- Two unloading trains
- Mixed loading/unloading scenario

Each instance contains:
- Train information
- Truck arrival information
- Container task information
- Yard positions
- AYC configuration

---

# Training

To train the proposed co-evolutionary reinforcement learning model:

```bash
python experiments/run_v14_coevo_rgcd.py
```

During training:

1. SAC learns scheduling policies.
2. Policy trajectories are collected.
3. Candidate scheduling rules are generated.
4. Rules are validated and selected.
5. Validated rules guide future policy optimization.

---

# Evaluation

To evaluate the trained model:

```bash
python experiments/run_rule_guided_evaluate.py
```

Evaluation metrics:

## Makespan

Weighted train completion time.

## Total Truck Waiting Time

Total waiting time caused by:
- Transfer-point availability
- AYC service delay

## AYC Workload Imbalance

Measures workload distribution among automated yard cranes.

## Runtime

Online decision-making computation time.
