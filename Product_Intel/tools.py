# product_intel/tools.py
# ----------------------
# All tool wiring lives here:
# - Playwright browser (for robust web interactions)
# - File management tools (sandboxed)
# - Push notifications (Pushover)
# - Web search (Serper), Wikipedia
# - Python REPL
# - Firecrawl (search + crawl) if available
#
# Each tool is wrapped as a LangChain Tool, so LangGraph's ToolNode can use them.

import os
import asyncio
import requests
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Playwright (async) for controlled browsing:
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

load_dotenv(override=True)

# --- Optional: Firecrawl (from agno.tools) ---
FIRECRAWL_AVAILABLE = False
try:
    from agno.tools.firecrawl import FirecrawlTools  # type: ignore
    FIRECRAWL_AVAILABLE = True
except Exception:
    FIRECRAWL_AVAILABLE = False

# --- Pushover for push notifications ---
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

# --- Serper (Google Serp API) ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")


async def playwright_tools() -> Tuple[List[Tool], object, object]:
    """
    Start Playwright and expose its browser tools via LangChain.
    Returns (tools, browser, playwright_controller) so caller can clean up.
    """
    playwright = await async_playwright().start()
    # headless=False for visibility while developing; turn to True in servers
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


def push(text: str) -> str:
    """
    Send a push notification via Pushover (best-effort).
    """
    if not (PUSHOVER_TOKEN and PUSHOVER_USER):
        return "pushover not configured"
    try:
        requests.post(
            PUSHOVER_URL,
            data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER, "message": text},
            timeout=10,
        )
        return "success"
    except Exception as e:
        return f"pushover error: {e}"


def get_file_tools() -> List[Tool]:
    """
    Safe file operations inside a sandbox directory.
    """
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


def get_search_tools() -> List[Tool]:
    """
    Serper (web search), Wikipedia query, and Python REPL for quick calcs.
    """
    tools: List[Tool] = []

    # Serper
    if SERPER_API_KEY:
        serper = GoogleSerperAPIWrapper()
        tools.append(
            Tool(
                name="search",
                func=serper.run,
                description="Search the web via Serper. Ideal for fresh info and links.",
            )
        )

    # Wikipedia
    wiki_api = WikipediaAPIWrapper()
    wiki = WikipediaQueryRun(api_wrapper=wiki_api)
    tools.append(
        Tool(
            name="wikipedia",
            func=wiki.run,
            description="Query Wikipedia for background information.",
        )
    )

    # Python REPL
    tools.append(PythonREPLTool())

    return tools


def get_push_tool() -> Tool:
    return Tool(
        name="send_push_notification",
        func=push,
        description="Send a push notification to the user (Pushover).",
    )


def get_firecrawl_tools() -> List[Tool]:
    """
    Optional Firecrawl (search + crawl). If unavailable, return [].
    """
    if not FIRECRAWL_AVAILABLE:
        return []

    fc = FirecrawlTools(search=True, crawl=True, poll_interval=10)
    return [
        Tool(
            name="firecrawl_search",
            func=fc.search,
            description="Search competitor/product info using Firecrawl.",
        ),
        Tool(
            name="firecrawl_crawl",
            func=fc.crawl,
            description="Crawl pages deeply for richer signals using Firecrawl.",
        ),
    ]


async def load_all_tools() -> Tuple[List[Tool], object, object]:
    """
    Compose all tools and initialize Playwright.
    Returns (tools, browser, playwright_instance)
    """
    pw_tools, browser, pw = await playwright_tools()
    tools = []
    tools += pw_tools
    tools += get_file_tools()
    tools += get_search_tools()
    tools += [get_push_tool()]
    tools += get_firecrawl_tools()  # optional

    return tools, browser, pw


async def graceful_close(browser, playwright):
    """
    Close Playwright resources, handling both running and non-running loops.
    """
    if not browser and not playwright:
        return
    try:
        loop = asyncio.get_running_loop()
        if browser:
            loop.create_task(browser.close())
        if playwright:
            loop.create_task(playwright.stop())
    except RuntimeError:
        # No running loop (e.g., shutdown hook)
        if browser:
            asyncio.run(browser.close())
        if playwright:
            asyncio.run(playwright.stop())
