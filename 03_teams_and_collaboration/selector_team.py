"""Multi-agent collaboration using model-driven dynamic speaker selection."""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    try:
        backend_dev = AssistantAgent(
            name="Backend_Engineer",
            model_client=model_client,
            system_message="You handle database design and API architecture. Answer technical backend questions.",
        )

        devops_eng = AssistantAgent(
            name="DevOps_Engineer",
            model_client=model_client,
            system_message="You handle Docker containers and CI/CD pipelines. Answer deployment questions.",
        )

        tech_lead = AssistantAgent(
            name="Tech_Lead",
            model_client=model_client,
            system_message=(
                "You orchestrate discussions, synthesize plans, and conclude reviews. "
                "When both backend and deployment plans are clear, say 'TERMINATE'."
            ),
        )

        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

        team = SelectorGroupChat(
            participants=[backend_dev, devops_eng, tech_lead],
            model_client=model_client,
            termination_condition=termination,
        )

        await Console(team.run_stream(task="Propose a minimal deployment setup for a high-traffic FastAPI application."))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
