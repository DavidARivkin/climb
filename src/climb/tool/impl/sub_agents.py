from typing import Any, Dict

from openai import AzureOpenAI, OpenAI

from climb.common import Session


def create_llm_client(
    session: Session,
    additional_kwargs_required: Dict[str, Any],
) -> Any:
    if session.engine_name in ("azure_openai_v1",):
        client = AzureOpenAI(
            azure_endpoint=additional_kwargs_required["azure_endpoint"],
            api_version=additional_kwargs_required["api_version"],
            api_key=additional_kwargs_required["api_key"],
        )
    else:
        # All OpenAI-compatible providers (OpenAI, Groq, Anthropic, Google, Ollama, etc.)
        client = OpenAI(
            api_key=additional_kwargs_required["api_key"],
            base_url=additional_kwargs_required.get("base_url"),
        )
    return client


def get_llm_chat(
    client: Any,
    session: Session,
    additional_kwargs_required: Dict[str, Any],
    chat_kwargs: Dict,
) -> str:
    from climb.engine._providers import get_provider

    if session.engine_name in ("azure_openai_v1",):
        model_type = additional_kwargs_required["azure_openai_config"].model
        out = client.chat.completions.create(
            model=additional_kwargs_required["azure_openai_config"].deployment_name,
            max_tokens=MODEL_MAX_MESSAGE_TOKENS[model_type],
            # ---
            messages=chat_kwargs["messages"],
            stream=chat_kwargs["stream"],
        )
    else:
        # All OpenAI-compatible providers
        provider_id = additional_kwargs_required.get("provider_id", "openai")
        provider = get_provider(provider_id)
        model_type = additional_kwargs_required["engine_params"]["model_id"]
        out = client.chat.completions.create(
            model=model_type,
            max_tokens=provider.get_max_message_tokens(model_type),
            temperature=additional_kwargs_required["engine_params"]["temperature"],
            # ---
            messages=chat_kwargs["messages"],
            stream=chat_kwargs["stream"],
        )
    out_text = out.choices[0].message.content  # type: ignore
    return out_text


# Keep backward compatibility reference.
from climb.engine.const import MODEL_MAX_MESSAGE_TOKENS  # noqa: E402
