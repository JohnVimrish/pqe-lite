"""
Centralized config. Phase 2 change: nothing is hardcoded in the
modules anymore -- every tunable value (connection details, model
thresholds, logging level) comes from here, which reads environment
variables with sensible local-dev defaults.

This matters for a portfolio project specifically because it's the
difference between "a script that works on my machine" and "a system
someone else could configure and run" -- worth being able to point to
in an interview as a deliberate choice, not an afterthought.
"""

from __future__ import annotations

import os
from dataclasses import dataclass



def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default




@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.environ.get("PQE_PG_HOST", "localhost")
    port: int = _env_int("PQE_PG_PORT", 5432)
    dbname: str = os.environ.get("PQE_PG_DB", "ai_experiment")
    user: str = os.environ.get("PQE_PG_USER", "user_one")
    password: str = os.environ.get("PQE_PG_PASSWORD", "")
    min_pool_size: int = _env_int("PQE_PG_POOL_MIN", 1)
    max_pool_size: int = _env_int("PQE_PG_POOL_MAX", 10)
    # TPC-H sample data typically lives outside the default "public"
    # schema (e.g. ai_ml_experiment) -- every pooled connection sets
    # search_path to this on open, see db.get_pool's configure callback.
    schema: str = os.environ.get("PQE_PG_SCHEMA", "ai_ml_experiment")
    statement_timeout_ms: int = _env_int("PQE_PG_STATEMENT_TIMEOUT_MS", 120_000)

    @property
    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


@dataclass(frozen=True)
class MongoConfig:
    uri: str = os.environ.get("PQE_MONGO_URI", "mongodb://admin:secretpassword@localhost:27017/?authSource=admin")
    db_name: str = os.environ.get("PQE_MONGO_DB", "pqe_lite")
    collection_name: str = os.environ.get("PQE_MONGO_COLLECTION", "decision_log")


@dataclass(frozen=True)
class ClassifierConfig:
    # Chosen empirically via scripts/train_classifier.py's
    # choose_threshold(); this default is a placeholder until you've
    # trained on real historical data. Never set this from a median.
    high_risk_threshold: float = _env_float("PQE_HIGH_RISK_THRESHOLD", 0.65)
    expensive_cost_threshold: float = _env_float(
        "PQE_EXPENSIVE_COST_THRESHOLD", 100_000.0
    )
    stale_seconds_threshold: float = _env_float(
        "PQE_STALE_SECONDS_THRESHOLD", 3600.0
    )
    model_path: str = os.environ.get("PQE_MODEL_PATH", "models/classifier.joblib")
    # Off by default -- see model_evaluation.py's build_estimator()
    # docstring for the leakage trap this avoids by construction, and
    # README's SMOTE section for why real data collection beats this
    # as your actual N-31 bottleneck fix.
    use_smote: bool = os.environ.get("PQE_USE_SMOTE", "false").lower() == "true"


@dataclass(frozen=True)
class LLMConfig:


    def normalize_nim_base_url(value: str | None) -> str:
        """
        ChatOpenAI expects the OpenAI-compatible API root, not the full
        /chat/completions endpoint. Accept either shape so .env is forgiving.
        """
        url = (value).strip().rstrip("/")
        completions_suffix = "/chat/completions"
        if url.endswith(completions_suffix):
            return url[: -len(completions_suffix)]
        return url


    default_nim_base_url: str = os.environ.get("PQE_LLM_DEFAULT_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    invoke_url: str = os.environ.get("PQE_LLM_NIM_INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    model: str = os.environ.get("PQE_LLM_MODEL", "google/gemma-4-31b-it")
    max_tokens: int = _env_int("PQE_LLM_MAX_TOKENS", 2048)
    max_retries: int = _env_int("PQE_LLM_MAX_RETRIES", 2)
    api_key: str = os.environ.get("NVIDIA_NIM_API_KEY")
    base_url: str = normalize_nim_base_url( default_nim_base_url or invoke_url )
    temperature: float = _env_float("NVIDIA_NIM_TEMPERATURE", 1.0)
    streaming: bool = _env_bool("NVIDIA_NIM_STREAM", False)
    top_p: float = _env_float("NVIDIA_NIM_TOP_P", 0.95)
   
    api_key: str  = ""
    
    model_kwargs = dict()

    # This is useful for supported reasoning models, but keep it env-controlled
    # so changing models later does not require changing code.
    if _env_bool("PQE_LLM_ENABLE_THINKING", False):
        model_kwargs["chat_template_kwargs"] = {"enable_thinking": True}


@dataclass(frozen=True)
class AppConfig:
    postgres: PostgresConfig = PostgresConfig()
    mongo: MongoConfig = MongoConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    llm: LLMConfig = LLMConfig()
    log_level: str = os.environ.get("PQE_LOG_LEVEL", "INFO")


def load_config() -> AppConfig:
    return AppConfig()
