"""End-to-end multi-agent news desk workflow: Web Search -> Fact Extraction -> Editorial -> Polish."""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from ddgs import DDGS

load_dotenv()


def search_latest_headlines(topic: str) -> str:
    """Fetches breaking news articles and snippets for a given topic or region.

    Args:
        topic (str): The search query or news topic.
    """
    try:
        results = DDGS().text(f"latest news headlines {topic}", max_results=3)
        if not results:
            return "No news results found."
        return "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in results])
    except Exception as e:
        return f"Search error: {str(e)}"


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    try:
        # Agent 1: Gathers live web data
        field_reporter = AssistantAgent(
            name="Field_Reporter",
            model_client=model_client,
            tools=[search_latest_headlines],
            reflect_on_tool_use=True,
            system_message=(
                "You are a field reporter. Use search_latest_headlines to retrieve raw facts. "
                "Summarize 2 key facts and pass them to the News_Anchor."
            ),
        )

        # Agent 2: Produces final bulletin and terminates
        news_anchor = AssistantAgent(
            name="News_Anchor",
            model_client=model_client,
            system_message=(
                "You are the senior news anchor. Transform the reporter's findings into a concise, "
                "punchy 2-sentence broadcast bulletin. End your message with 'TERMINATE'."
            ),
        )

        team = RoundRobinGroupChat(
            participants=[field_reporter, news_anchor],
            termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(5),
        )

        await Console(team.run_stream(task="Get the latest technology news headlines."))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
