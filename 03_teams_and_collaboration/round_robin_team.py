"""Multi-agent collaboration using deterministic RoundRobinGroupChat."""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
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
        # Agent 1: Researches and outlines core facts
        researcher = AssistantAgent(
            name="researcher",
            model_client=model_client,
            system_message="You are a technical researcher. State 3 key technical facts on the topic concisely.",
        )

        # Agent 2: Reviews, summarizes, and signals completion
        editor = AssistantAgent(
            name="editor",
            model_client=model_client,
            system_message=(
                "You are an editor. Convert the researcher's facts into a concise 2-sentence executive summary. "
                "End your response with the single word 'TERMINATE' once complete."
            ),
        )

        # Termination condition: terminate on keyword or maximum turns
        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(4)

        team = RoundRobinGroupChat(
            participants=[researcher, editor],
            termination_condition=termination,
        )

        await Console(team.run_stream(task="Explain the utility of asyncio event loops in modern Python."))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
