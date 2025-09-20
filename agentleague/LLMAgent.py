#  traces at https://platform.openai.com/logs?api=traces
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Callable

from agents import Agent
from agents import function_tool
from agents import OpenAIChatCompletionsModel
from agents import Runner
from agents import trace
from agents.tool import FunctionTool
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from agenttools.email_sender import send_html_email
from agenttools.web_search import web_search

# Add the project root to Python path and import agenttools
sys.path.insert(0, str(Path(__file__).parent.parent))


# Load environment variables
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)


# Create the email tool outside the class to avoid 'self' parameter issues
@function_tool
def send_html_email_tool(
    sender: str | None = 'engineer.atique@gmail.com',
    to: str | None = 'engineer.atique@gmail.com',
    subject: str | None = 'Test Email from LLMAgent',
    html_body: str | None = '<p>This is a test email from LLMAgent.</p>',
) -> str:
    """Sends an HTML email using tools/email_sender.

    Args:
        sender: The email address of the sender.
        to: The email address of the recipient.
        subject: The subject of the email.
        html_body: The HTML content of the email.

    Returns:
        A confirmation message indicating the email was sent.
    """
    logging.info(f"Sending email to {to} with subject '{subject}'")
    return send_html_email(
        to=to,
        sender=sender,
        subject=subject,
        html_body=html_body,
    )


# Using tools/web_search.py to create web searching tool for LLM Agents
@function_tool
def search_web(query: str, num_results: int = 2) -> dict:
    return web_search(query=query, num_results=num_results)


class LLMAgent(BaseModel):
    """LLM agent class for managing agent configurations and tools."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(FunctionTool,),
    )

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
    tools: list[Any] = Field(
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

    @staticmethod
    def timer(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator that logs the execution time of a function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                func_name = func.__name__
                class_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else 'Unknown'
                logging.info(f"{class_name}.{func_name} executed in {execution_time:.4f} seconds")
        return wrapper

    @timer
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


async def main():
    # Example usage
    model = os.getenv('OLLAMA_BASE_MODEL') or 'gpt-oss:20b'
    async_openai_client = AsyncOpenAI(
        base_url=os.getenv('OLLAMA_BASE_URL'),
        api_key=os.getenv('OLLAMA_API_KEY'),
    )
    agent_model = OpenAIChatCompletionsModel(model=model, openai_client=async_openai_client)

    agent = Agent(
        name='emailer_agent',
        instructions='You are an expert agent that can search the web and send emails by using the available tools.',
        model=agent_model,
        tools=[search_web, send_html_email_tool],
    )
    sender, to = 'engineer.atique@gmail.com', 'engineer.atique@gmail.com'
    prompt = (
        'Can you search for latest tech news and events and send it by email? '
        f"Use sender address {sender} and recipient address {to}"
        '. Make the email rhetorical, comical, funny, satirical and full of humor.'
    )
    with trace(f"{datetime.datetime.now().isoformat()}: Emailer {model}"):
        await Runner.run(agent, prompt)

if __name__ == '__main__':
    asyncio.run(main())
