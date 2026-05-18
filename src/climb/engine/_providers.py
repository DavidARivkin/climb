"""LLM provider registry.

Each provider defines how to connect to an OpenAI-compatible chat completions API.
Most providers reuse the `openai` SDK with a different `base_url`.
"""

import os
from typing import Dict, List, Optional

import pydantic


class ProviderConfig(pydantic.BaseModel):
    """Configuration for an LLM provider."""

    name: str  # Display name, e.g. "OpenAI"
    provider_id: str  # Machine key, e.g. "openai"
    env_var_api_key: str  # e.g. "OPENAI_API_KEY"
    default_base_url: Optional[str] = None  # None = use SDK default
    allowed_models: List[str]
    default_model: str
    model_max_message_tokens: Dict[str, int]
    model_context_size: Dict[str, int]
    supports_streaming_usage: bool = True
    requires_api_key: bool = True

    def get_max_message_tokens(self, model_id: str) -> int:
        return self.model_max_message_tokens.get(model_id, 4_096)

    def get_context_size(self, model_id: str) -> int:
        return self.model_context_size.get(model_id, 128_000)


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

PROVIDERS: Dict[str, ProviderConfig] = {
    # --- OpenAI (default) ---
    "openai": ProviderConfig(
        name="OpenAI",
        provider_id="openai",
        env_var_api_key="OPENAI_API_KEY",
        default_base_url=None,
        allowed_models=[
            "gpt-5-2025-08-07",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4o-mini-2024-07-18",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
        ],
        default_model="gpt-4o-2024-11-20",
        model_max_message_tokens={
            "gpt-5-2025-08-07": 128_000,
            "gpt-4o-2024-11-20": 16_384,
            "gpt-4o-2024-08-06": 16_384,
            "gpt-4o-2024-05-13": 4_096,
            "gpt-4-turbo-2024-04-09": 4_096,
            "gpt-4-1106-preview": 4_096,
            "gpt-4-0125-preview": 4_096,
            "gpt-4o-mini-2024-07-18": 16_384,
            "gpt-3.5-turbo-1106": 4_096,
            "gpt-3.5-turbo-0125": 4_096,
        },
        model_context_size={
            "gpt-5-2025-08-07": 272_000,
            "gpt-4o-2024-11-20": 128_000,
            "gpt-4o-2024-08-06": 128_000,
            "gpt-4o-2024-05-13": 128_000,
            "gpt-4-turbo-2024-04-09": 128_000,
            "gpt-4-1106-preview": 128_000,
            "gpt-4-0125-preview": 128_000,
            "gpt-4o-mini-2024-07-18": 128_000,
            "gpt-3.5-turbo-1106": 16_385,
            "gpt-3.5-turbo-0125": 16_385,
        },
    ),
    # --- Anthropic ---
    "anthropic": ProviderConfig(
        name="Anthropic",
        provider_id="anthropic",
        env_var_api_key="ANTHROPIC_API_KEY",
        default_base_url="https://api.anthropic.com/v1",
        allowed_models=[
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ],
        default_model="claude-sonnet-4-20250514",
        model_max_message_tokens={
            "claude-sonnet-4-20250514": 16_384,
            "claude-haiku-4-5-20251001": 8_192,
            "claude-3-5-sonnet-20241022": 8_192,
            "claude-3-5-haiku-20241022": 8_192,
        },
        model_context_size={
            "claude-sonnet-4-20250514": 200_000,
            "claude-haiku-4-5-20251001": 200_000,
            "claude-3-5-sonnet-20241022": 200_000,
            "claude-3-5-haiku-20241022": 200_000,
        },
        supports_streaming_usage=False,
    ),
    # --- Google ---
    "google": ProviderConfig(
        name="Google",
        provider_id="google",
        env_var_api_key="GOOGLE_API_KEY",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        allowed_models=[
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ],
        default_model="gemini-2.5-pro-preview-05-06",
        model_max_message_tokens={
            "gemini-2.5-pro-preview-05-06": 16_384,
            "gemini-2.5-flash-preview-05-20": 16_384,
            "gemini-2.0-flash": 8_192,
            "gemini-2.0-flash-lite": 4_096,
        },
        model_context_size={
            "gemini-2.5-pro-preview-05-06": 1_048_576,
            "gemini-2.5-flash-preview-05-20": 1_048_576,
            "gemini-2.0-flash": 1_048_576,
            "gemini-2.0-flash-lite": 1_048_576,
        },
    ),
    # --- Groq ---
    "groq": ProviderConfig(
        name="Groq",
        provider_id="groq",
        env_var_api_key="GROQ_API_KEY",
        default_base_url="https://api.groq.com/openai/v1",
        allowed_models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        default_model="llama-3.3-70b-versatile",
        model_max_message_tokens={
            "llama-3.3-70b-versatile": 16_384,
            "llama-3.1-8b-instant": 8_192,
            "mixtral-8x7b-32768": 4_096,
            "gemma2-9b-it": 8_192,
        },
        model_context_size={
            "llama-3.3-70b-versatile": 128_000,
            "llama-3.1-8b-instant": 128_000,
            "mixtral-8x7b-32768": 32_768,
            "gemma2-9b-it": 8_192,
        },
    ),
    # --- Ollama (local) ---
    "ollama": ProviderConfig(
        name="Ollama (local)",
        provider_id="ollama",
        env_var_api_key="OLLAMA_API_KEY",
        default_base_url="http://localhost:11434/v1",
        allowed_models=[
            "llama3.1",
            "llama3.1:70b",
            "mistral",
            "codellama",
            "qwen2.5",
            "gemma2",
        ],
        default_model="llama3.1",
        model_max_message_tokens={
            "llama3.1": 4_096,
            "llama3.1:70b": 4_096,
            "mistral": 4_096,
            "codellama": 4_096,
            "qwen2.5": 4_096,
            "gemma2": 4_096,
        },
        model_context_size={
            "llama3.1": 128_000,
            "llama3.1:70b": 128_000,
            "mistral": 32_000,
            "codellama": 16_000,
            "qwen2.5": 128_000,
            "gemma2": 8_192,
        },
        supports_streaming_usage=False,
        requires_api_key=False,
    ),
    # --- Z.ai (Zyphra) ---
    "zai": ProviderConfig(
        name="Z.ai (Zyphra)",
        provider_id="zai",
        env_var_api_key="ZAI_API_KEY",
        default_base_url="https://api.zyphra.com/v1",
        allowed_models=[
            "Zyphra-1",
        ],
        default_model="Zyphra-1",
        model_max_message_tokens={
            "Zyphra-1": 16_384,
        },
        model_context_size={
            "Zyphra-1": 128_000,
        },
    ),
    # --- DeepSeek ---
    "deepseek": ProviderConfig(
        name="DeepSeek",
        provider_id="deepseek",
        env_var_api_key="DEEPSEEK_API_KEY",
        default_base_url="https://api.deepseek.com",
        allowed_models=[
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        default_model="deepseek-chat",
        model_max_message_tokens={
            "deepseek-chat": 8_192,
            "deepseek-reasoner": 8_192,
        },
        model_context_size={
            "deepseek-chat": 128_000,
            "deepseek-reasoner": 128_000,
        },
    ),
    # --- Together AI ---
    "together": ProviderConfig(
        name="Together AI",
        provider_id="together",
        env_var_api_key="TOGETHER_API_KEY",
        default_base_url="https://api.together.xyz/v1",
        allowed_models=[
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_max_message_tokens={
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": 4_096,
            "mistralai/Mixtral-8x7B-Instruct-v0.1": 4_096,
            "Qwen/Qwen2.5-72B-Instruct-Turbo": 4_096,
        },
        model_context_size={
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": 128_000,
            "mistralai/Mixtral-8x7B-Instruct-v0.1": 32_768,
            "Qwen/Qwen2.5-72B-Instruct-Turbo": 128_000,
        },
    ),
    # --- OpenRouter ---
    "openrouter": ProviderConfig(
        name="OpenRouter",
        provider_id="openrouter",
        env_var_api_key="OPENROUTER_API_KEY",
        default_base_url="https://openrouter.ai/api/v1",
        allowed_models=[
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4-20250514",
            "google/gemini-2.5-pro-preview",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        default_model="openai/gpt-4o",
        model_max_message_tokens={
            "openai/gpt-4o": 16_384,
            "anthropic/claude-sonnet-4-20250514": 16_384,
            "google/gemini-2.5-pro-preview": 16_384,
            "meta-llama/llama-3.3-70b-instruct": 4_096,
        },
        model_context_size={
            "openai/gpt-4o": 128_000,
            "anthropic/claude-sonnet-4-20250514": 200_000,
            "google/gemini-2.5-pro-preview": 1_048_576,
            "meta-llama/llama-3.3-70b-instruct": 128_000,
        },
    ),
    # --- Generic OpenAI-compatible ---
    "openai_compatible": ProviderConfig(
        name="OpenAI-compatible (custom)",
        provider_id="openai_compatible",
        env_var_api_key="OPENAI_COMPAT_API_KEY",
        default_base_url=None,
        allowed_models=[
            "default",
        ],
        default_model="default",
        model_max_message_tokens={
            "default": 4_096,
        },
        model_context_size={
            "default": 128_000,
        },
        supports_streaming_usage=False,
    ),
}


def get_provider(provider_id: Optional[str] = None) -> ProviderConfig:
    """Return provider config.

    If provider_id is None, reads LLM_PROVIDER env var (default: "openai").
    """
    if provider_id is None:
        provider_id = os.environ.get("LLM_PROVIDER", "openai")
    if provider_id not in PROVIDERS:
        raise ValueError(f"Unknown provider: '{provider_id}'. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider_id]


def get_all_models_for_provider(provider_id: str) -> List[str]:
    """Return allowed models for a provider."""
    provider = PROVIDERS[provider_id]
    return provider.allowed_models
