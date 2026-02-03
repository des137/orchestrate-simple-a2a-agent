import logging
import os
import sys

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from app.agent import CalculatorGreetingAgent
from app.agent_executor import CalculatorGreetingAgentExecutor


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=8080)
def main(host, port):
    """Starts the Calculator & Greeting Agent server."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)

        # Define calculator skill
        calculator_skill = AgentSkill(
            id="calculator",
            name="Calculator",
            description="Performs basic arithmetic operations: add, subtract, multiply, and divide",
            tags=["calculator", "math", "arithmetic"],
            examples=[
                "What is 25 plus 17?",
                "Calculate 8 times 9",
                "Divide 144 by 12",
                "Add 10 and 5, then multiply the result by 3",
            ],
        )

        # Define greeting skill
        greeting_skill = AgentSkill(
            id="greeting",
            name="Greeting Generator",
            description="Generates friendly personalized greetings",
            tags=["greeting", "hello", "introduction"],
            examples=["Hi, my name is Alice!", "Say hello to Bob", "Greet John"],
        )

        agent_card = AgentCard(
            name="Calculator & Greeting Agent",
            description="A helpful assistant that can greet people and perform basic arithmetic calculations",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=CalculatorGreetingAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=CalculatorGreetingAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[calculator_skill, greeting_skill],
        )

        # Initialize A2A server components
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client, config_store=push_config_store
        )
        request_handler = DefaultRequestHandler(
            agent_executor=CalculatorGreetingAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        logger.info(f"Starting Calculator & Greeting Agent on {host}:{port}")
        logger.info(
            f"Agent Card available at: http://{host}:{port}/.well-known/agent.json"
        )
        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
