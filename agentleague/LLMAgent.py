#  traces at https://platform.openai.com/logs?api=traces
from __future__ import annotations

import asyncio
import datetime
import logging
import os
from typing import Any

from agents import Agent
from agents import OpenAIChatCompletionsModel
from agents import Runner
from agents import trace
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
# Load environment variables
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)


class LLMAgent(BaseModel):
    """LLM agent class for managing agent configurations and tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        default='LLMAgent',
        description='The name of the agent.',
    )
    description: str = Field(
        default='An expert agent that utilizes provided model and various tools for AI tasks.',
        description="A brief description of the agent's purpose.",
    )
    base_url: str | None = Field(
        default=None,
        description="The base URL for the agent's API.",
    )
    api_key: str | None = Field(
        default=None,
        description='The API key for authenticating requests.',
    )
    base_model: str = Field(
        default='gpt-4',
        description='The base model to use for the agent.',
    )
    openai_client: AsyncOpenAI | None = Field(
        default=None,
        description='The OpenAI client for interacting with the OpenAI API.',
        exclude=True,  # Exclude from serialization since it's not JSON serializable
    )
    openai_chat_completion_model: Any = Field(
        default=None,
        description='The OpenAI model instance for chat completions.',
        exclude=True,  # Exclude from serialization since it's not JSON serializable
    )
    instructions: str | None = Field(
        default=None,
        description='Initial instructions for the agent.',
    )
    core_agent: Any = Field(
        default=None,
        description='The core agent instance.',
        exclude=True,  # Exclude from serialization since it's not JSON serializable
    )
    core_agent_as_tool: Any = Field(
        default=None,
        description='The core agent represented as a tool.',
        exclude=True,  # Exclude from serialization since it's not JSON serializable
    )
    tools: list[str] = Field(
        default_factory=list,
        description='A list of tools the agent can use.',
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.openai_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self.openai_chat_completion_model = OpenAIChatCompletionsModel(
            model=self.base_model,
            openai_client=self.openai_client,
        )
        self.core_agent = Agent(
            name='core_agent',
            instructions=self.instructions or 'You are a helpful assistant that uses tools to assist with various tasks.',
            model=self.openai_chat_completion_model,
        )
        self.core_agent_as_tool = self.core_agent.as_tool(
            tool_name='core_agent_as_tool',
            tool_description=f"A tool for the {self.name}.",
        )
        logging.info(f"Initialized LLMAgent with name: {self.name}, base_model: {self.base_model}")
        logging.info(f"Available instructions: {self.instructions}")

    def get_core_agent(self) -> Agent:
        """Returns the core agent instance."""
        return self.core_agent

    def get_core_agent_as_tool(self) -> Any:
        """Returns the core agent represented as a tool."""
        return self.core_agent_as_tool

    def ask_agent(self, prompt: str) -> str:
        """Ask the core agent a question and get a response.

        Args:
            prompt: The question or prompt to send to the agent
        """
        async def _ask_agent(prompt: str) -> str:
            try:
                result = await Runner.run(self.core_agent, prompt)
                # Access the final_output attribute which contains the agent's response
                if hasattr(result, 'final_output'):
                    return str(result.final_output)
                else:
                    logging.error(f"RunResult missing final_output attribute. Available: {dir(result)}")
                    raise AttributeError("RunResult object doesn't have 'final_output' attribute")
            except Exception as e:
                logging.error(f"Error running agent with prompt '{prompt}': {e}")
                raise
        with trace(f'LLMAgent {self.name} @ {datetime.datetime.now().isoformat()}'):
            response = asyncio.run(_ask_agent(prompt))
        return response


if __name__ == '__main__':
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
