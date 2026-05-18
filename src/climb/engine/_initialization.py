from typing import Any, Dict

from climb.common.data_structures import Session
from climb.common.utils import fix_windows_path_backslashes
from climb.db import DB

from ._azure_config import (
    AZURE_OPENAI_CONFIG_PATH,
    get_api_key_for_azure_openai,
    load_azure_openai_config_item,
)
from ._config import get_dotenv_config
from ._engine import EngineBase
from ._providers import get_provider
from .engine_openai_v1 import AzureOpenAIV1Engine, OpenV1Engine

dotenv_config = get_dotenv_config()


ENGINE_MAP = {
    # v1 engines:
    OpenV1Engine.get_engine_name(): OpenV1Engine,
    AzureOpenAIV1Engine.get_engine_name(): AzureOpenAIV1Engine,
}

azure_engines = [engine_name for engine_name in ENGINE_MAP.keys() if "azure" in engine_name]
non_azure_engines = [engine_name for engine_name in ENGINE_MAP.keys() if "azure" not in engine_name]


def create_engine(db: DB, session: Session, config: Dict[str, Any]) -> EngineBase:
    EngineClass = ENGINE_MAP[session.engine_name]
    conda_path = config.get("CONDA_PATH", None)
    if conda_path is not None:
        conda_path = fix_windows_path_backslashes(conda_path)
    extra_kwargs: Dict[str, Any] = {
        "conda_path": conda_path,
    }
    if EngineClass.get_engine_name() in non_azure_engines:
        # Determine provider from session params or env var.
        provider_id = session.engine_params.get("provider_id", None)
        if provider_id is None:
            provider_id = config.get("LLM_PROVIDER", "openai")
        provider = get_provider(provider_id)

        # Resolve API key: session param override > env var > empty.
        api_key = session.engine_params.get("api_key_override", None)
        if not api_key:
            api_key = config.get(provider.env_var_api_key, "")
        extra_kwargs["api_key"] = api_key
        # Base URL: session param override > provider default > None.
        base_url = session.engine_params.get("base_url_override", None)
        if not base_url:
            base_url = provider.default_base_url
        extra_kwargs["base_url"] = base_url
        extra_kwargs["provider_id"] = provider_id
    elif EngineClass.get_engine_name() in azure_engines:
        extra_kwargs["azure_openai_config"] = load_azure_openai_config_item(
            AZURE_OPENAI_CONFIG_PATH,
            session.engine_params["config_item_name"],  # pyright: ignore
        )
        extra_kwargs["api_key"] = get_api_key_for_azure_openai(extra_kwargs["azure_openai_config"], config)
    return EngineClass(
        db=db,
        session=session,
        **extra_kwargs,
    )
