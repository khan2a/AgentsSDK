# simple example showing how to create an agent that tells jokes
# traces go to https://platform.openai.com/traces
from __future__ import annotations

import asyncio

from agents import Agent
from agents import Runner
from agents import trace
from dotenv import load_dotenv


async def test_jokes_agent():
    load_dotenv(override=True)
    with trace('test_jokes_agent'):
        agent = Agent(
            name='Jokes Agent',
            model='gpt-4o-mini',
            instructions='Tell a joke about the topic provided by the user.',
        )
        result = await Runner.run(agent, 'chickens')
        return result.final_output

if __name__ == '__main__':
    print(asyncio.run(test_jokes_agent()))
