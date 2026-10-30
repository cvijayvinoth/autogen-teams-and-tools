"""Composing conditional termination rules in AutoGen v0.4+."""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    TokenUsageTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
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
            name="counter_agent",
            model_client=model_client,
            system_message="You count numbers upwards starting from 1. Say 'COMPLETE' when you reach 3.",
        )

        # Composable termination: triggers on keyword OR safety limit of 5 turns
        stop_on_word = TextMentionTermination("COMPLETE")
        stop_on_limit = MaxMessageTermination(5)
        combined_termination = stop_on_word | stop_on_limit

        team = RoundRobinGroupChat(
            participants=[agent],
            termination_condition=combined_termination,
        )

        await Console(team.run_stream(task="Count to 3, one message per number."))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
