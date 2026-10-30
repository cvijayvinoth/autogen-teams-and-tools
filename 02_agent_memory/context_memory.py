"""Demonstrates multi-turn memory preservation within an agent session."""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    try:
        agent = AssistantAgent(
            name="context_agent",
            model_client=model_client,
            system_message="You are a helpful assistant that tracks user preferences across turns.",
        )

        # Turn 1: Setting context
        print("=== Turn 1: Storing Preferences ===")
        await Console(agent.run_stream(task="My name is Vijay and I work with Python in Bangalore."))

        # Turn 2: Querying from internal session memory
        print("\n=== Turn 2: Recalling Stored Context ===")
        await Console(agent.run_stream(task="What programming language and city did I mention earlier? Answer in 1 line."))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
