# PyCRM

[![CI](https://github.com/TristanBester/counting-reward-machines/actions/workflows/ci.yaml/badge.svg)](https://github.com/TristanBester/counting-reward-machines/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/TristanBester/counting-reward-machines/graph/badge.svg?token=NBFYD2O05M)](https://codecov.io/gh/TristanBester/counting-reward-machines)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://pycrm.xyz)
[![arXiv](https://img.shields.io/badge/arXiv-2312.11364-b31b1b.svg)](https://arxiv.org/abs/2312.11364)

A Python framework for formal task specification and efficient reinforcement learning with **Reward Machines (RMs)** and **Counting Reward Machines (CRMs)**.

[Documentation](https://pycrm.xyz) | [Paper](https://arxiv.org/abs/2312.11364) | [Quick Start](#quick-start)

## Overview

PyCRM provides a unified framework for **Reward Machines (RMs)** and **Counting Reward Machines (CRMs)**, offering a formal approach to reward specification in reinforcement learning. RMs handle regular tasks with finite-state automata, while CRMs extend this with counters for Turing-complete expressiveness, enabling efficient learning through structured reward functions and counterfactual experiences.

## Features

- **Unified RM/CRM Support**: First-class support for both Reward Machines and Counting Reward Machines
- **Reinforcement Learning Integration**: Ready-to-use agents that leverage counterfactual experiences
- **Cross-Product Environments**: Framework for combining ground environments with RMs or CRMs
- **Modular Design**: Composable automata for complex task specifications
- **Expressive Power**: From regular languages (RMs) to Turing-complete specifications (CRMs)
- **Example Environments**: Complete worked examples for both RMs and CRMs

## Quick Start

### Installation

```bash
pip install pyrewardmachines
```

For detailed installation instructions and troubleshooting, see the [Installation Guide](https://pycrm.xyz/installation).

### Basic Usage

See the [Quick Start Guide](https://pycrm.xyz/quickstart) for complete examples of creating and using both Reward Machines and Counting Reward Machines, including:

- Setting up ground environments, labelling functions, and automata (RMs or CRMs)
- Creating cross-product environments
- Training agents with counterfactual experiences

For a comprehensive introduction to the framework, see the [Introduction](https://pycrm.xyz/introduction).

## Key Components

The PyCRM framework consists of several key components:

- **Ground Environment**: The base environment (typically a Gymnasium environment)
- **Labelling Function**: Maps environment observations to symbolic events
- **Automaton**: Formal specification of the task (either a Reward Machine or Counting Reward Machine)
- **Cross-Product Environment**: Combines all components into a learning environment
- **RL Agents**: Algorithms that leverage counterfactual experiences for improved sample efficiency

For detailed explanations of these components, see the [Core Concepts](https://pycrm.xyz/core-concepts) section in the documentation.

## Applications

- **Task-Oriented RL**: Specify complex objectives with structured reward functions
- **Robotics**: Define temporally extended tasks with symbolic events
- **Formal Verification**: Guarantee task completion through CRM properties
- **Curriculum Learning**: Progressively build task complexity

For complete worked examples demonstrating these applications, see the [Worked Examples](https://pycrm.xyz/worked-examples) section in the documentation.

## Citation

If you use Counting Reward Machines in your research, please cite:

```bibtex
@article{bester2023counting,
  title={Counting Reward Automata: Sample Efficient Reinforcement Learning Through the Exploitation of Reward Function Structure},
  author={Bester, Tristan and Rosman, Benjamin and James, Steven and Tasse, Geraud Nangue},
  journal={arXiv preprint arXiv:2312.11364},
  year={2023}
}
```

## Contributing

Contributions are welcome. To get started:

```bash
# Clone repository
git clone https://github.com/TristanBester/pycrm.git
cd pycrm

# Set up virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run comprehensive testing across environments
uv pip install tox
uv run tox
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
