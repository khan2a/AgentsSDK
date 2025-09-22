# https://platform.openai.com/traces
# cd /Users/AKhan2/github.com/khan2a/AgentsSDK
# source .venv/bin/activate
# python examples/deep_research.py
from __future__ import annotations

import asyncio
import logging
import time as time_module
from datetime import time

from agents import Agent
from agents import function_tool
from agents import ModelSettings
from agents import Runner
from agents import RunResult
from agents import trace
from agents import WebSearchTool
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field

from agenttools.email_sender import send_html_email

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEPARATOR = '\n=======================================\n'
HOW_MANY_SEARCHES = 3
sender, recipient = 'engineer.atique@gmail.com', 'engineer.atique@gmail.com'
todays_date = f"Remember, today is {time_module.strftime('%B %d, %Y')}."

topic = f'Quantum Computing as of {todays_date}'

instructions_for_search_planner_agent = (
    'You are a research planner agent. '
    'Given a query, come up with a set of web searches to perform to best answer the query. '
    'Output {HOW_MANY_SEARCHES} terms to query for.'
    f' {todays_date} so all searches should be relevant to current events.'
)

instructions_for_web_search_agent = (
    'You are a research assistant. '
    'Your task is to search the web for the given topic and gather information on it and provide a comprehensive summary. '
    'The summary should include key points, relevant data, and any important findings. '
    'Make sure to cite your sources and provide links to the original articles or papers. '
    'Your summary should be clear, concise, and well-structured. '
    'The summary must be two to three paragraphs long, containing less than 300 words. '
    'This summary will be used to generate a report, so ensure it is informative and accurate.'
    f' {todays_date} so all search results should be from the last 6 months.'
)

instructions_for_emailer_agent = (
    'You are an agent that can generate and send professional html emails. '
    'You will be provided with short summary, main report, and follow up questions. You should use your tool to generate a single email. '
    'The email should be professional and well formatted in HTML. '
    'HTML should have visually appealing and aesthetic formatting, with appropriate headings, paragraphs, and lists where necessary. '
    'The email should include a brief introduction, the summary of the research, and a closing statement. '
    'You should use your tool to send one email, providing the report converted into clean, well presented HTML. '
    'Always include all the references at the bottom of the email. '
    'Use the clue from subject which year is the report about. '
    f' {todays_date} so all references should be from the last 6 months.'

)

instructions_for_senior_researcher_agent = (
    'You are a senior researcher. '
    'You will be provided with a set of web search results. '
    'Your task is to analyze the results and provide a comprehensive report of the findings. '
    'The report should include key points, relevant data, and any important findings. '
    'Make sure to cite your sources and provide links to the original articles or papers. '
    'Your report should be clear, concise, and well-structured. '
    'The report must be comprehensive in its coverage of the topic. You can decide the length of the report. '
    'This report will be used to generate an email, so ensure it is informative and accurate.'
    f' {todays_date} so all references should be from the last 6 months.'
)


class WebSearchItem(BaseModel):
    reason: str = Field(..., description='Your reasoning for why this search is important to the query.')
    query: str = Field(..., description='The search term to use for the web search.')


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(..., description=f'A list of {HOW_MANY_SEARCHES} search items.')


async def _run_agent(starting_agent: Agent, input: str) -> RunResult:
    with trace(input):
        run_result = await Runner.run(starting_agent=starting_agent, input=input)
    return run_result


class ReportData(BaseModel):
    short_summary: str = Field(..., description='A brief summary of the research topic.')
    report: str = Field(..., description='The final report.')
    follow_up_questions: list[str] = Field(..., description='Suggested topics to research further.')


@function_tool
def send_html_email_tool(
    sender: str,
    to: str,
    subject: str,
    html_body: str,
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


async def plan_searches(topic: str) -> WebSearchPlan:
    """Plans a series of web searches to perform on a given topic."""
    search_planner_agent = Agent(
        name='SearchPlannerAgent',
        instructions=instructions_for_search_planner_agent,
        model='gpt-4o-mini',
        output_type=WebSearchPlan,
    )
    result = await Runner.run(starting_agent=search_planner_agent, input=topic)
    return result.final_output


async def perform_web_search(search_plan: WebSearchPlan) -> list[str]:
    """Performs a web search using the WebSearchTool."""
    web_search_agent = Agent(
        name='WebSearchAgent',
        instructions=instructions_for_web_search_agent,
        tools=[WebSearchTool(search_context_size='low')],
        model='gpt-4o-mini',
        model_settings=ModelSettings(tool_choice='required'),
    )

    async def _search(item: WebSearchItem) -> str:
        search_input = f"search item: {item.query}\nreason for searching: {item.reason}"
        result = await Runner.run(starting_agent=web_search_agent, input=search_input)
        return result.final_output

    tasks = [asyncio.create_task(_search(item)) for item in search_plan.searches]
    search_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and return only successful results
    valid_results = [result for result in search_results if isinstance(result, str)]
    return valid_results


async def generate_report(topic: str, search_results: list[str]) -> ReportData:
    """Generates a report based on the web search results with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            senior_researcher_agent = Agent(
                name='SeniorResearcherAgent',
                instructions=instructions_for_senior_researcher_agent,
                model='gpt-4o-mini',
                output_type=ReportData,
            )
            prompt = (
                f"Original query: {topic}\nSummarized search results: {search_results}"
            )
            result = await Runner.run(starting_agent=senior_researcher_agent, input=prompt)
            return result.final_output
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info('Retrying in 2 seconds...')
                await asyncio.sleep(2)
            else:
                logging.error('All retry attempts failed. Creating fallback report.')
                # Create a fallback report manually
                combined_results = '\n\n'.join(search_results)
                fallback_report = (
                    f"# Research Report: {topic}\n\n"
                    '## Executive Summary\n'
                    'Due to technical difficulties, this report contains the raw search results:\n\n'
                    f"{combined_results}"
                )
                return ReportData(
                    short_summary=f"Research on {topic} faced API connectivity issues.",
                    report=fallback_report,
                    follow_up_questions=[
                        'What are the main drivers of negative CPaaS market trends?',
                        'How can CPaaS providers mitigate revenue leakage?',
                        'What security measures are most critical for CPaaS platforms?',
                    ],
                )

    # This should never be reached due to the fallback in the except block
    raise RuntimeError('Unexpected error in generate_report')


async def send_email(report: ReportData) -> str:
    """Sends the generated report via email."""
    email_agent = Agent(
        name='EmailAgent',
        instructions=instructions_for_emailer_agent,
        tools=[send_html_email_tool],
        model='gpt-4o-mini',
        model_settings=ModelSettings(tool_choice='required'),
    )
    prompt = (
        f"Short Summary: {report.short_summary}\n"
        f"Main Report: {report.report}\n"
        f"Follow Up Questions: {report.follow_up_questions}\n"
        f"Recipient: {recipient}\n"
        f"Sender: {sender}\n"
        f"Subject: Research Report on {topic}\n"
    )
    result = await Runner.run(starting_agent=email_agent, input=prompt)
    return result.final_output


if __name__ == '__main__':
    logging.info(SEPARATOR)
    logging.info(f"Starting deep research on topic: {topic}")
    search_plan = asyncio.run(plan_searches(topic=topic))
    logging.info(f"Search Plan: {search_plan.model_dump_json(indent=2)}")
    time_module.sleep(2)  # To ensure logs are in order

    logging.info(SEPARATOR)
    logging.info('Performing web searches...')
    search_results = asyncio.run(perform_web_search(search_plan=search_plan))
    logging.info(f"Search Results: {search_results}")
    time_module.sleep(2)  # To ensure logs are in order

    logging.info(SEPARATOR)
    logging.info('Generating report from search results...')
    report = asyncio.run(generate_report(topic=topic, search_results=search_results))
    logging.info(f"Generated Report: {report.model_dump_json(indent=2)}")
    time_module.sleep(2)  # To ensure logs are in order

    logging.info(SEPARATOR)
    logging.info('Sending report via email...')
    email_result = asyncio.run(send_email(report=report))
    logging.info(f"Email Result: {email_result}")
    logging.info('Deep research process completed.')
