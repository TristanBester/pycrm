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

Counting Reward Machines (CRMs) are formal models that combine the expressive power of reward machines with counter mechanisms inspired by formal language theory. This framework provides a principled way to:

- Specify complex, temporally-extended tasks
- Generate counterfactual learning experiences
- Dramatically improve reinforcement learning efficiency
- Enable interpretable reward structures

CRMs define rewards based on symbolic events and their counts, allowing for elegant specification of tasks that would be cumbersome to express with standard reward functions.

## ✨ Features

- 🤖 **Reinforcement Learning Integration**: Ready-to-use agents that leverage counterfactual experiences
- 🔄 **Cross-Product Environments**: Combine ground environments with CRMs to create learning tasks
- 🏗️ **Modular Design**: Easily compose CRMs for complex task specifications
- 📊 **Expressive Power Hierarchy**: Regular, context-free, and context-sensitive CRMs
- 🧪 **Example Environments**: Complete worked examples for quick understanding
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
    crm=SimpleCRM(),
    lf=your_labelling_function
)

# Train with counterfactual experiences
agent = CounterfactualQLearningAgent(env=cross_product)
agent.learn(total_episodes=1000)
```

## 🔍 Hierarchy of CRMs

CRMs come in three main variants, each with increasing expressive power:

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
- **RL Agents**: Algorithms that leverage counterfactual experiences

## 📊 Performance

Counterfactual experience generation enables dramatic improvements in learning efficiency:

| Environment | Standard Q-Learning | Counterfactual Q-Learning | Speedup |
|-------------|---------------------|---------------------------|---------|
| Letter World | ~4500 episodes | ~400 episodes | 11.25x |
| Warehouse | ~8000 episodes | ~650 episodes | 12.30x |
| Minecraft | ~12000 episodes | ~800 episodes | 15.00x |

## 📋 Citation

If you use Counting Reward Machines in your research, please cite:

```bibtex
@article{neary2023counting,
  title={Counting Reward Machines: Expressivity and Counterfactual Experience Generation},
  author={Neary, Cyrus and Bester, Tristan and Brafman, Ronen I and Desai, Andrey and Tamar, Aviv},
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
pytest

# Run tox for testing across environments
uv pip install tox
tox
```

Tox is used to ensure compatibility across multiple Python versions and environments. It runs the test suite, linting, and type checking all at once.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The project builds on research from formal language theory and reward machines
- Thanks to all contributors and the reinforcement learning community
- Special thanks to our research partners and supporting institutions