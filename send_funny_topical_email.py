# traces go to https://platform.openai.com/traces
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from agents import Agent
from agents import function_tool
from agents import Runner
from agents import trace
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from tools import email_sender
from tools.web_search import web_search_tool

MY_EMAIL = 'engineer.atique@gmail.com'


# define email sending tool
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
        email_sender.send_email(sender, recipient, subject, body)
        logger.info('Email sent successfully to %s', recipient)
        return {'status': 'success', 'message': 'Email sent successfully.'}
    except Exception as e:
        logger.exception('Failed to send email to %s: %s', recipient, e)
        return {'status': 'error', 'message': str(e)}


# Define agents with specific instructions
humorous_email_instructions = (
    'You write humorous emails on any topic given by the user.'
    ' The email should be funny and engaging.'
    ' Use randon, current events and popular culture to make the email topical and relevant.'
    ' Make sure to include a catchy subject line that aligns with the topics discussed.'
    ' To fetch current topics and information, use the web_search tool.'
    ' The web_search tool will provide you several recent articles on current topics.'
    ' Use the provided articles to make the email more relevant and interesting.'
    ' You can incorporate jokes, puns, or witty remarks related to the topic.'
    ' Also to include the provided topis, make the email newsletter style.'
    ' Use sentences like "According to a recent article from [source], ..." to reference the articles.'
    ' Make sure to add all the topsics provided by the web_search tool.'
    ' Change paragraphs and use sentences like "In other news, ..." to transition between topics.'
    f' Remember todays date is {datetime.now().strftime("%B %d, %Y")}.'
    ' Use the above date to make the email relevant to current events.'
    ' Also use the above date in the email content to give an impression of timeliness.'
    ' Add links to relevant articles if possible.'
)
humorous_email_agent = Agent(
    name='Humorous Email Agent',
    model='gpt-4o-mini',
    instructions=humorous_email_instructions,
    tools=[web_search_tool],
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
    ' The email should be clear and concise.'
)
serious_email_agent = Agent(
    name='Serious Email Agent',
    model='gpt-4o-mini',
    instructions=serious_email_instructions,
)

email_selector_instructions = (
    'select the best email from the given options based on user prompt.'
    ' Respond with the best email only. Do not add any extra text.'
)
email_selector_agent = Agent(
    name='Email Selector Agent',
    model='gpt-4o-mini',
    instructions=email_selector_instructions,
)

# define list of tools including email sending tool and agents as tools
humorous_email_agent_as_tool = humorous_email_agent.as_tool(
    tool_name='humorous_email_tool',
    tool_description='Writes a humorous email on a given topic.',
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
            logger.info('Humorous Email %d generated (length=%d)', idx, len(output or ''))
            print('\nHumorous Email:\n', output)

    with trace('pick best humorous email'):
        email_selector_agent_input = 'select best humorous email from: ' + ('\n\n'.join(outputs))
        logger.info('Selecting best humorous email using selector agent')
        best_humorous_email = await asyncio.gather(
            Runner.run(starting_agent=email_selector_agent, input=email_selector_agent_input),
        )
        logger.info('Best humorous email selected (length=%d)', len(best_humorous_email[0].final_output or ''))
        print('\nBest Humorous Email:\n', best_humorous_email[0].final_output)


async def send_funny_topical_email(
    topic: str | None = None,
    recipient: str = MY_EMAIL,
        sender: str = MY_EMAIL,
) -> None:
    load_dotenv(override=True)

    if topic is None or topic.strip() == '':
        topic = (f'anything interesting and funny that is the talk of the town. Infotainment and trending stuff.'
                 f' in {datetime.now().strftime("%B, %Y")}')

    emailer_agent_input = (
        f'Write and send an humorous email to {recipient}.'
        f' Use the tools available to you to write and send the email.'
        f' Generate a funny email about topic {topic} for email content.'
        f' If the topic is openly defined, let the tools decide which topic to generate.'
        f' Using the web_search tool, find recent articles on the topic to include in the email.'
        f' Send the email from {sender}.'
    )
    tools = [humorous_email_agent_as_tool, send_email_tool]
    emailer_agent = Agent(
        name='Emailer Agent',
        model='gpt-4o-mini',
        instructions='You write and send emails using the tools available to you.',
        tools=tools,
    )

    with trace(f'use agents as tools @ {datetime.now().isoformat()}'):
        logger.info('Running emailer agent to write and send email: topic=%s recipient=%s', topic, recipient)
        result = await Runner.run(starting_agent=emailer_agent, input=emailer_agent_input)
        logger.info('Emailer agent finished. Result length=%d', len(result.final_output or ''))
        print('\nFinal Output:\n', result.final_output)

if __name__ == '__main__':
    recipient = MY_EMAIL
    topic = None
    logger.info('Starting main run: topic=%s recipient=%s', topic, recipient)
    asyncio.run(send_funny_topical_email(topic=topic, recipient=recipient))
