from __future__ import annotations

import asyncio

from agents import Agent
from agents import function_tool
from agents import Runner
from agents import trace
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from tools import email

humorous_email_instructions = (
    'You write humorous emails on any topic given by the user.'
    'The email should be funny and engaging.'
)
humorous_email_agent = Agent(
    name='Humorous Email Agent',
    model='gpt-4o-mini',
    instructions=humorous_email_instructions,
)

serious_email_instructions = (
    'You write serious and professional emails on any topic given by the user.'
    'The email should be clear and concise.'
)
serious_email_agent = Agent(
    name='Serious Email Agent',
    model='gpt-4o-mini',
    instructions=serious_email_instructions,
)

email_selector_instructions = (
    'select the best email from the given options based on user prompt.'
    'Respond with the best email only. Do not add any extra text.'
)
email_selector_agent = Agent(
    name='Email Selector Agent',
    model='gpt-4o-mini',
    instructions=email_selector_instructions,
)


async def run_streamed_email_agent(starting_agent: Agent, input: str) -> None:
    load_dotenv(override=True)
    with trace('run_streamed_email_agent'):
        result = Runner.run_streamed(
            starting_agent,
            input,
        )

        async for event in result.stream_events():
            if event.type == 'raw_response_event' and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end='', flush=True)


@function_tool
def send_email_tool(sender: str, recipient: str, subject: str, body: str) -> dict:
    """
    Sends an email using the email tool.
    Args:
        sender (str): The email address of the sender.
        recipient (str): The email address of the recipient.
        subject (str): The subject of the email.
        body (str): The body content of the email.
    Returns:
        dict: A dictionary indicating the status of the email sending operation and a message.
    """
    email.send_email(sender, recipient, subject, body)
    return {'status': 'success', 'message': 'Email sent successfully.'}


async def pick_the_best_humorous_email():
    load_dotenv(override=True)
    humorous_email_agent_input = 'Write brief email to customer about his missed payment.'
    with trace('parallel humorous emails'):
        humorous_emails = await asyncio.gather(
            Runner.run(starting_agent=humorous_email_agent, input=humorous_email_agent_input),
            Runner.run(starting_agent=humorous_email_agent, input=humorous_email_agent_input),
        )

        outputs = [humorous_email.final_output for humorous_email in humorous_emails]
        for output in outputs:
            print('\nHumorous Email:\n', output)

    with trace('pick best humorous email'):
        email_selector_agent_input = 'select best humorous email from: '+('\n\n'.join(outputs))
        best_humorous_email = await asyncio.gather(
            Runner.run(starting_agent=email_selector_agent, input=email_selector_agent_input),
        )
        print('\nBest Humorous Email:\n', best_humorous_email[0].final_output)

if __name__ == '__main__':
    # asyncio.run(pick_the_best_humorous_email())
    print(f'send_email_tool: {send_email_tool}')
