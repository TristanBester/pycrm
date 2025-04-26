# Counting Reward Machines

[![CI](https://github.com/TristanBester/counting-reward-machines/actions/workflows/ci.yaml/badge.svg)](https://github.com/TristanBester/counting-reward-machines/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/TristanBester/counting-reward-machines/graph/badge.svg?token=NBFYD2O05M)](https://codecov.io/gh/TristanBester/counting-reward-machines)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://crm.tristanbester.xyz)
[![arXiv](https://img.shields.io/badge/arXiv-2312.11364-b31b1b.svg)](https://arxiv.org/abs/2312.11364)

**A framework for formal task specification and efficient reinforcement learning with Counting Reward Machines**

[Documentation](https://crm.tristanbester.xyz) | [Paper](https://arxiv.org/abs/2312.11364) | [Demo](https://crm.tristanbester.xyz) | [Quick Start](#quick-start)

## 🌟 Overview

Counting Reward Machines (CRMs) are formal models that offer Turing-complete reward specification in RL. This framework provides a principled way to:

- Specify complex, temporally-extended tasks
- Generate counterfactual learning experiences
- Dramatically improve reinforcement learning efficiency
- Enable interpretable reward structures

## ✨ Features

- 🤖 **Reinforcement Learning Integration**: Ready-to-use agents that leverage counterfactual experiences
- 🔄 **Cross-Product Environments**: Combine ground environments with CRMs to create learning tasks
- 🏗️ **Modular Design**: Easily compose CRMs for complex task specifications
- 📊 **Expressive Power Hierarchy**: Regular, context-free, and context-sensitive CRMs
- 🧪 **Example Environments**: Complete worked examples to get you started
- 📝 **Comprehensive Documentation**: Detailed guides and API references

## 🚀 Quick Start

### Installation

```bash
pip install counting-reward-machines
```

### Basic Example

```python
from crm.automaton import CountingRewardMachine
from crm.agents.tabular.cql import CounterfactualQLearningAgent

# Define a Counting Reward Machine (simplified)
class SimpleCRM(CountingRewardMachine):
    # Implementation details...
    pass

# Create a cross-product environment
cross_product = create_cross_product_environment(
    ground_env=your_environment,
    lf=your_labelling_function
    crm=SimpleCRM(),
)

# Train with counterfactual experiences
agent = CounterfactualQLearningAgent(env=cross_product)
agent.learn(total_episodes=1000)
```

## 🔍 Chomsky Reward Hierarchy 

CRMs are able to model tasks at all levels of the Chomsky reward hierarchy. They can be used to specify tasks that are regular, context-free, or context-sensitive. This allows for a wide range of applications in reinforcement learning.

| Type | Description | Counter Logic | Example Task |
|------|-------------|---------------|--------------|
| Regular | Single counter | Simple counting | "Collect 3 items" |
| Context-Free | Multiple independent counters | Balance counting | "Match #A with #B" |
| Context-Sensitive | Multiple dependent counters | Complex relations | "If #A>3, then #B>#C" |

Each type enables specification of increasingly complex tasks while maintaining formal semantics.

## 💡 Use Cases

- 🎮 **Task-Oriented RL**: Specify complex objectives with natural language-like structures
- 🤖 **Robotics**: Define temporally extended tasks with rich symbolic events
- 🔍 **Formal Verification**: Guarantee task completion through CRM properties
- 🧠 **Curriculum Learning**: Progressively build task complexity by extending CRMs

## 📚 Key Components

The CRM framework consists of several key components:

- **Ground Environment**: The base environment (usually a Gymnasium env)
- **Labelling Function**: Maps environment observations to symbolic events
- **Counting Reward Machine**: Formal specification of the task
- **Cross-Product Environment**: Combines all components into a learning environment
- **RL Agents**: Algorithms that leverage counterfactual experiences for increased sample efficiency


## 📋 Citation

If you use Counting Reward Machines in your research, please cite:

```bibtex
@article{bester2023counting,
  title={Counting Reward Automata: Sample Efficient Reinforcement Learning Through the Exploitation of Reward Function Structure},
  author={Bester, Tristan and Rosman, Benjamin and James, Steven and Tasse, Geraud Nangue},
  journal={arXiv preprint arXiv:2312.11364},
  year={2023}
}
```

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Clone repository
git clone https://github.com/TristanBester/counting-reward-machines.git
cd counting-reward-machines

# Set up virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies with uv
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run tox for testing across environments
uv pip install tox
uv run tox
```

Tox is used to ensure compatibility across multiple Python versions and environments. It runs the test suite, linting, and type checking all at once.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
