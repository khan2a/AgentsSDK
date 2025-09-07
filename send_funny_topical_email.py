# traces go to https://platform.openai.com/traces
from __future__ import annotations

import asyncio
import logging

from agents import Agent
from agents import function_tool
from agents import Runner
from agents import trace
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from tools import email

MY_EMAIL = 'engineer.atique@gmail.com'

# Define agents with specific instructions
humorous_email_instructions = (
    'You write humorous emails on any topic given by the user.'
    'The email should be funny and engaging.'
)
humorous_email_agent = Agent(
    name='Humorous Email Agent',
    model='gpt-4o-mini',
    instructions=humorous_email_instructions,
)
logger = logging.getLogger(__name__)

# Configure basic logging. Users can override this by configuring logging before
# importing this module or by setting the LOGLEVEL environment variable.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger.info('Configured logging for main module')

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

# define list of tools including email sending tool and agents as tools
humorous_email_agent_as_tool = humorous_email_agent.as_tool(
    tool_name='humorous_email_tool',
    tool_description='Writes a humorous email on a given topic.'
)


async def run_streamed_email_agent(starting_agent: Agent, input: str) -> None:
    load_dotenv(override=True)
    with trace('run_streamed_email_agent'):
        logger.info('Starting streamed runner for agent %s with input: %s', starting_agent.name, input)
        result = Runner.run_streamed(
            starting_agent,
            input,
        )

        async for event in result.stream_events():
            logger.debug('Stream event received: %s', getattr(event, 'type', repr(event)))
            if event.type == 'raw_response_event' and isinstance(event.data, ResponseTextDeltaEvent):
                logger.info('Streaming response delta: %s', event.data.delta)
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
    logger.info('send_email_tool called: from=%s to=%s subject=%s', sender, recipient, subject)
    try:
        email.send_email(sender, recipient, subject, body)
        logger.info('Email sent successfully to %s', recipient)
        return {'status': 'success', 'message': 'Email sent successfully.'}
    except Exception as e:
        logger.exception('Failed to send email to %s: %s', recipient, e)
        return {'status': 'error', 'message': str(e)}


async def pick_the_best_humorous_email():
    load_dotenv(override=True)
    humorous_email_agent_input = 'Write brief email to customer about his missed payment.'
    with trace('parallel humorous emails'):
        logger.info('Generating multiple humorous emails in parallel')
        humorous_emails = await asyncio.gather(
            Runner.run(starting_agent=humorous_email_agent, input=humorous_email_agent_input),
            Runner.run(starting_agent=humorous_email_agent, input=humorous_email_agent_input),
        )

        outputs = [humorous_email.final_output for humorous_email in humorous_emails]
        for idx, output in enumerate(outputs, start=1):
            logger.info('Humorous Email %d generated (length=%d)', idx, len(output or ""))
            print('\nHumorous Email:\n', output)

    with trace('pick best humorous email'):
        email_selector_agent_input = 'select best humorous email from: ' + ('\n\n'.join(outputs))
        logger.info('Selecting best humorous email using selector agent')
        best_humorous_email = await asyncio.gather(
            Runner.run(starting_agent=email_selector_agent, input=email_selector_agent_input),
        )
        logger.info('Best humorous email selected (length=%d)', len(best_humorous_email[0].final_output or ""))
        print('\nBest Humorous Email:\n', best_humorous_email[0].final_output)


async def send_funny_topical_email(
    topic: str | None = None,
    recipient: str = MY_EMAIL,
        sender: str = MY_EMAIL
        ) -> None:
    load_dotenv(override=True)

    if topic is None or topic.strip() == '':
        topic = 'anything interesting and funny that is the talk of the town these days'

    emailer_agent_input = (
        f'Write and send an humorous email to {recipient}.'
        ' Use the tools available to you to write and send the email.'
        f' Generate a funny email about topic {topic} for email content.'
        ' Generate a funny subject line as well according to the topic'
        f' Send the email from {sender}.'
    )
    tools = [humorous_email_agent_as_tool, send_email_tool]
    emailer_agent = Agent(
        name='Emailer Agent',
        model='gpt-4o-mini',
        instructions='You write and send emails using the tools available to you.',
        tools=tools
    )

    with trace('use agents as tools'):
        logger.info('Running emailer agent to write and send email: topic=%s recipient=%s', topic, recipient)
        result = await Runner.run(starting_agent=emailer_agent, input=emailer_agent_input)
        logger.info('Emailer agent finished. Result length=%d', len(result.final_output or ""))
        print('\nFinal Output:\n', result.final_output)

if __name__ == '__main__':
    recipient = MY_EMAIL
    topic = None
    logger.info('Starting main run: topic=%s recipient=%s', topic, recipient)
    asyncio.run(send_funny_topical_email(topic=topic, recipient=recipient))
