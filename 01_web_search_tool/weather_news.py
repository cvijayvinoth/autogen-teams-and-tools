"""Day 2 Assignment: AutoGen Weather News Agent with DDGS tool integration."""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from ddgs import DDGS

load_dotenv()


def search_weather_news(city: str) -> str:
    """Searches the web for today's live weather news, forecasts, and climate alerts for a given city.

    Do not use this tool for general trivia, historical events, or non-weather inquiries.

    Args:
        city (str): The name of the city or location (e.g., 'Chennai', 'London', 'Tokyo').

    Returns:
        str: Raw search results containing recent weather news snippets.
    """
    query = f"today current weather forecast news alerts {city}"
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
            return f"No recent weather news found for {city}."
        return "\n".join([f"- {item.get('title')}: {item.get('body')}" for item in results])
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    try:
        agent = AssistantAgent(
            name="weather_news_agent",
            model_client=model_client,
            tools=[search_weather_news],
            reflect_on_tool_use=True,
            system_message=(
                "You are a weather news assistant. Use the search_weather_news tool "
                "to retrieve up-to-date weather updates. Always answer the user in "
                "exactly 3 short lines summarizing the current conditions or news."
            ),
        )

        city = input("Enter city name: ").strip()
        if not city:
            print("No city provided. Exiting.")
            return

        task = f"Provide today's weather news update for {city}. Reply in exactly 3 short lines."
        await Console(agent.run_stream(task=task))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
