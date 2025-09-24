"""Tests for the shared Pydantic base model used by the agent."""

from __future__ import annotations

import pytest
from pydantic import Field, ValidationError

from common import AgentBaseModel


class Person(AgentBaseModel):
    """Example structured response schema."""

    name: str
    age: int


def test_agent_base_model_enforces_types() -> None:
    result = Person(name="Ada", age=37)

    assert result.name == "Ada"
    assert result.age == 37

    with pytest.raises(ValidationError):
        Person(name="Ada", age="37")


def test_agent_base_model_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Person(name="Grace", age=35, nickname="hopper")


def test_agent_base_model_populate_by_name() -> None:
    class AliasExample(AgentBaseModel):
        """Model using aliases but still accepting field names."""

        display_name: str = Field(alias="displayName")

    result = AliasExample(display_name="Curie")

    assert result.display_name == "Curie"
    assert result.model_dump(by_alias=True) == {"displayName": "Curie"}
