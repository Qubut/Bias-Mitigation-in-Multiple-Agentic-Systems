# Bias Mitigation in Multi-Agent Systems (MAS)

## Overview

**Bias-Mitigation-in-Multiple-Agentic-Systems** is a research-oriented project focused on identifying, analyzing, and mitigating biases within multi-agent systems. This repository provides a complete development environment with tooling for reproducibility, testing, and deployment of bias mitigation strategies in agentic AI systems.

## 🚀 Quick Start

### Prerequisites

- [Devenv](https://devenv.sh/getting-started/)
- [Docker](https://docs.docker.com/get-docker/) or [Podman](https://podman.io/docs/installation)

### Installation

1. **Clone and enter the project:**

```bash
git clone https://github.com/Qubut/Bias-Mitigation-in-Multiple-Agentic-Systems/
cd bias-mitigation-in-mas
```

2. **Start the development environment:**

```bash
devenv shell # or direnv allow if you also already installed direnv
```

3. **Start required services:**

```bash
docker-compose up  # Starts Ollama, MongoDB, and Neo4j
# or
podman-compose up
```

### Project Structure

```
bias-mitigation-in-mas/
├── packages/bias-mitigation/   # Main Python package
│   ├── src/                    # Source code
│   ├── test/                   # Test files
│   ├── devenv.nix             # Nix environment configuration
│   ├── pyproject.toml         # Python dependencies and tooling
│   └── docker-compose.yml     # Service containers
├── devenv/                     # Global devenv configuration
├── README.md                   # This file
└── modules/python.nix         # Python-specific Nix module
```

## 🛠️ Development Environment

This project uses a reproducible development environment with:

### Core Tools

- **Python 3.12** with UV for package management
- **Ruff & Black** for code formatting and linting
- **Mypy** for static type checking
- **Pytest** for testing with asyncio support

### Services (via Docker Compose)

- **Ollama** (port `11434`): Local LLM inference with GPU support
- **MongoDB** (port `27017`): Document database with Beanie ODM
- **Neo4j** (port `7687/7474`): Graph database for knowledge representation

### Development Commands

```bash
# Enter development shell
devenv shell

# Run tests
pytest

# Format code
black .
ruff check --fix

# Type checking
mypy src/

# Start Jupyter notebook
devenv up jupyter
```

## 📦 Dependencies

Key Python dependencies include:

- **Agent Frameworks**: LangChain, DSPy for agent orchestration
- **Data Processing**: Polars, Sentence Transformers for embeddings
- **Databases**: Beanie (MongoDB ODM), Neo4j driver
- **Memory Management**: Mem0 for agent memory
- **Configuration**: OmegaConf, DotMap
- **Async Utilities**: aiofiles, asyncio, returns

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# MongoDB
MONGO_ROOT_USER=admin
MONGO_ROOT_PASSWORD=secure_password
MONGO_PORT=27017

# Neo4j
NEO4J_PASSWORD=neo4j_password
NEO4J_BOLT_PORT=7687
NEO4J_BROWSER_PORT=7474

# Ollama
OLLAMA_PORT=11434
```

### Editor Integration

**VS Code/Pyright**: Uses `pyrightconfig.json` for type checking and path configuration.

**Jupyter**: Configuration is stored in `.jupyter/` with notebook server on port `8888`.

## 🧪 Testing

The project includes a comprehensive testing setup:

```bash
# Run all tests
pytest

# Run specific test module
pytest test/test_bias_mitigation/

# Run with coverage
pytest --cov=src
```

## 📚 Documentation

Documentation follows Google-style docstrings and can be generated using:

```bash
# Check documentation quality
ruff check --select D

# Generate API documentation (requires pdoc or similar)
uv run pdoc src/bias_mitigation -o docs/
```

## 🔄 Git Hooks

Pre-commit hooks are configured for:

- Code formatting (Black)
- Linting (Ruff)
- Type checking (Mypy)
- Shell script validation (ShellCheck)

## 🧠 Research Focus

This project investigates bias in multi-agent systems through:

1. **Bias Detection**: Identifying biases in agent decision-making
2. **Mitigation Strategies**: Implementing fairness-aware algorithms
3. **Evaluation Framework**: Metrics and benchmarks for bias measurement
4. **Reproducible Experiments**: Containerized environments for consistent results

## 🤝 Contributing

1. Ensure you have the development environment set up
2. Create a feature branch from `main`
3. Make changes with appropriate tests
4. Run `black`, `ruff`, and `mypy` before committing
5. Submit a pull request with a clear description

## 📄 License

This project is part of research initiatives. Please contact the maintainers for licensing information.

## 🆘 Support

For issues with:

- **Development environment**: Check Devenv documentation
- **Service containers**: Check Docker logs and healthchecks
- **Testing**: Review pytest configuration in `pyproject.toml`

---

*This project is maintained as part of ongoing research in AI fairness and multi-agent systems.*
