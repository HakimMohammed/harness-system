import logging

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from rich import print

from .config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # Initialize the chat model with the specified provider and model
    try:
        model: BaseChatModel = init_chat_model(
            model_provider="mistralai", model="mistral-small-latest", configurable_fields=["api_key"]
        )
    except ValueError:
        logger.error("Provider is not supported by Langchain")
        return
    except ImportError:
        logger.error("Provider's integration package is not installed.")
        return

    # Create the agent using the initialized model, add tools, system prompt ....
    agent: CompiledStateGraph = create_agent(model=model)

    # Interact with the agent
    messages = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ]
    }
    ## Provide api key on runtime
    config = {
        "configurable": {
            "api_key": settings.mistral_api_key,
        }
    }

    try:
        result = agent.invoke(messages, config=config)
    except ValueError:
        logger.error("Provider is not supported by Langchain")
        return
    except ImportError:
        logger.error("Provider's integration package is not installed.")
        return
    print(result)
    # print(f"[bold blue]Agent response:[/bold blue] {result['messages'][0]['content']}")
