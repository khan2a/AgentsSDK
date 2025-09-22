# AgentsSDK
Using OpenAI Agent SDK for Agent Control Flows, building Tools and Applications

## Environment Setup

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv self update
uv venv .venv
source .venv/bin/activate
uv init
uv add --dev ruff flake8
ruff check .
flake8 .
uv pip install pre-commit
pre-commit
pre-commit run --all-files
uv sync
uv pip install -r requirements.txt
```

## Example Scripts

### agentleague/LLMAgent.py

**Description:**
`LLMAgent.py` demonstrates how to build and run a custom LLM agent using the OpenAI Agent SDK. It integrates tools such as web search and email sending, and shows how to structure agent workflows, use guardrails, and trace agent actions. This script is ideal for experimenting with agent capabilities, tool integration, and custom logic.

**Usage:**
Activate your virtual environment and run the script from the project root:
```bash
source .venv/bin/activate
python agentleague/LLMAgent.py
```
You can view agent traces at [OpenAI traces dashboard](https://platform.openai.com/logs?api=traces).

---

### examples/deep_research.py

**Description:**
`deep_research.py` is a workflow example for automated research using LLM agents. It plans web searches, executes them, summarizes findings, and generates a professional report. The script demonstrates multi-agent orchestration, error handling, and integration with external tools (web search, email). It is useful for research automation and report generation.

**Usage:**
Activate your virtual environment and run the script from the project root:
```bash
source .venv/bin/activate
python examples/deep_research.py
```
The script will output search plans, results, and a final report. You can customize the research topic in the script.

---
