"""Shared Pydantic base classes for structured agent outputs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AgentBaseModel(BaseModel):
    """Base model for structured outputs returned by LangGraph agents.

    This class centralizes common configuration for schemas that are exposed to
    LLM tool responses, ensuring consistent serialization and validation rules.
    Downstream models should inherit from this base rather than directly from
    ``pydantic.BaseModel``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid", strict=True)


__all__ = ["AgentBaseModel"]
