"""Web search tool for AgentsSDK.

This module provides a `web_search` function exported as a tool via
the `@function_tool` decorator so agents can call it. It tries these
methods in order:

- SerpAPI (if SERPAPI_API_KEY is present)
- Bing Web Search (if BING_API_KEY and BING_ENDPOINT are present)
- DuckDuckGo HTML scraping as a last-resort fallback

Configure API keys through environment variables. Example:

export SERPAPI_API_KEY="..."
export BING_API_KEY="..."
export BING_ENDPOINT="https://api.bing.microsoft.com/v7.0/search"

The tool returns a JSON-serializable dict describing results.
"""
from __future__ import annotations

import html
import logging
import os

import requests  # type: ignore[import]
from agents import function_tool
from bs4 import BeautifulSoup  # type: ignore[import]
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)


def _serpapi_search(query: str, num: int = 5) -> list[dict]:
    """Search using SerpAPI (https://serpapi.com).

    Returns a list of result dicts with keys: title, snippet, link, source.
    """
    load_dotenv(override=True)
    api_key = os.getenv('SERPAPI_API_KEY')
    if not api_key:
        raise RuntimeError('SERPAPI_API_KEY not set')

    params = {
        'q': query,
        'api_key': api_key,
        'num': num,
        'engine': 'google',
    }
    resp = requests.get('https://serpapi.com/search', params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get('organic_results', [])[:num]:
        results.append({
            'title': item.get('title'),
            'snippet': item.get('snippet') or item.get('snippet_highlighted') or '',
            'link': item.get('link'),
            'source': 'serpapi',
        })
    return results


def _bing_search(query: str, num: int = 5) -> list[dict]:
    """Search using Bing Web Search API (Azure/Bing)."""
    api_key = os.getenv('BING_API_KEY')
    endpoint = os.getenv('BING_ENDPOINT')
    if not api_key or not endpoint:
        raise RuntimeError('BING_API_KEY or BING_ENDPOINT not set')

    headers = {'Ocp-Apim-Subscription-Key': api_key}
    params = {'q': query, 'count': num}
    resp = requests.get(endpoint, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get('webPages', {}).get('value', [])[:num]:
        results.append({
            'title': item.get('name'),
            'snippet': item.get('snippet'),
            'link': item.get('url'),
            'source': 'bing',
        })
    return results


def _duckduckgo_search(query: str, num: int = 5) -> list[dict]:
    """Fallback scraper for DuckDuckGo search results page.

    This is a simple HTML scrape and is fragile. Use only when API
    keys are unavailable.
    """
    resp = requests.get('https://html.duckduckgo.com/html/', params={'q': query}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    results = []
    for a in soup.select('a.result__a')[:num]:
        title = a.get_text(strip=True)
        link = a.get('href')
        # DuckDuckGo often wraps links; best-effort unescape
        link = html.unescape(str(link)) if link else ''
        parent_div = a.find_parent('div', class_='result')
        snippet_el = None
        if parent_div and hasattr(parent_div, 'select_one') and callable(getattr(parent_div, 'select_one', None)):
            snippet_el = parent_div.select_one('.result__snippet')  # type: ignore
        snippet = snippet_el.get_text(strip=True) if snippet_el else ''
        results.append({'title': title, 'snippet': snippet, 'link': link, 'source': 'duckduckgo'})
    return results


def _web_search_impl(query: str, num_results: int = 5, source: str = 'serpapi') -> dict:
    """Internal implementation of web search.

    This function is a plain callable. A FunctionTool wrapper is created
    below as `web_search_tool` for use by agents.
    """
    logger.info('web_search called: query=%s num_results=%d source=%s', query, num_results, source)

    engines_tried = []
    # Resolve engine order
    preferred = []
    if source != 'auto':
        preferred = [source]
    else:
        preferred = ['serpapi', 'bing', 'duckduckgo']

    last_exc = None
    for engine in preferred:
        try:
            engines_tried.append(engine)
            if engine == 'serpapi':
                results = _serpapi_search(query, num_results)
            elif engine == 'bing':
                results = _bing_search(query, num_results)
            elif engine == 'duckduckgo':
                results = _duckduckgo_search(query, num_results)
            else:
                raise RuntimeError(f'Unknown engine: {engine}')
            logger.info('web_search using engine=%s returned %d results', engine, len(results))
            return {'engine': engine, 'results': results}
        except Exception as e:  # pragma: no cover - simple runtime fallback logic
            logger.debug('web_search engine %s failed: %s', engine, e, exc_info=True)
            last_exc = e

    # If all engines failed, raise a clear error with hints
    msg = f'All web search backends failed. Tried: {engines_tried}. Last error: {last_exc}'
    logger.error(msg)
    return {'engine': None, 'results': [], 'error': str(last_exc)}


# Public callable for other modules to call directly
def web_search(query: str, num_results: int = 5, source: str = 'serpapi') -> dict:
    return _web_search_impl(query=query, num_results=num_results, source=source)


# FunctionTool wrapper for the agents framework
web_search_tool = function_tool(_web_search_impl)


if __name__ == '__main__':
    query = 'Latest news on AI advancements'
    result = _serpapi_search(query, num=3)
    print(result)
