# cd agentssdk
# source .venv/bin/activate
# python -m examples.test_agent
from __future__ import annotations

import os

from dotenv import load_dotenv

from agentleague.LLMAgent import LLMAgent
load_dotenv(override=True)

name = 'GeminiAgent'
base_url = os.getenv('GEMINI_BASE_URL')
api_key = os.getenv('GEMINI_API_KEY')
base_model = os.getenv('GEMINI_BASE_MODEL')
instructions = (
    'You are an expert assistant that can perform a variety of tasks using the tools at your disposal. '
    'Use the tools wisely to assist with user requests.'
)
gemini_agent = LLMAgent(name=name, base_url=base_url, api_key=api_key, base_model=base_model, instructions=instructions)
prompt = 'What is the capital of France?'
print(f"Prompt: {prompt}")
response = gemini_agent.ask_agent(prompt)
print(f"Response: {response}")

name = 'GroqAgent'
base_url = os.getenv('GROQ_BASE_URL')
api_key = os.getenv('GROQ_API_KEY')
base_model = os.getenv('GROQ_BASE_MODEL')
groq_agent = LLMAgent(name=name, base_url=base_url, api_key=api_key, base_model=base_model, instructions=instructions)
prompt = 'What is the capital of Germany?'
print(f"Prompt: {prompt}")
response = groq_agent.ask_agent(prompt)
print(f"Response: {response}")
