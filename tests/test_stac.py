"""Unit tests for src/STAC.py — validation logic and pure functions.

These tests cover JSON validation in BaseLM, Generator, Judge, and Planner
without requiring API keys or model access.

Run:  python -m pytest tests/test_stac.py -v
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# BaseLM.check_valid_json
# ---------------------------------------------------------------------------

class TestBaseLMCheckValidJson:
    """Test the base JSON validation used by all STAC pipeline classes."""

    def _make_validator(self, fields):
        """Create a minimal object with check_valid_json from BaseLM."""
        # Import inline to avoid triggering model initialization
        from src.STAC import BaseLM
        obj = object.__new__(BaseLM)
        obj.output_json_fields = fields
        return obj

    def test_none_is_invalid(self):
        v = self._make_validator(["field"])
        assert v.check_valid_json(None) is False

    def test_missing_field_is_invalid(self):
        v = self._make_validator(["a", "b"])
        assert v.check_valid_json({"a": 1}) is False

    def test_all_fields_present(self):
        v = self._make_validator(["a", "b"])
        assert v.check_valid_json({"a": 1, "b": 2}) is True

    def test_empty_fields_always_valid(self):
        v = self._make_validator([])
        assert v.check_valid_json({"anything": True}) is True

    def test_extra_fields_ok(self):
        v = self._make_validator(["a"])
        assert v.check_valid_json({"a": 1, "b": 2, "c": 3}) is True


# ---------------------------------------------------------------------------
# Generator.check_valid_json
# ---------------------------------------------------------------------------

class TestGeneratorCheckValidJson:
    def _make_validator(self):
        from src.STAC import Generator
        obj = object.__new__(Generator)
        obj.output_json_fields = ["tool_name", "parameters"]
        return obj

    def test_valid_tool_chain_attacks(self):
        v = self._make_validator()
        data = {
            "tool_chain_attacks": [
                {
                    "attack_goal": "steal data",
                    "tool_chain": [
                        {"tool_name": "send_email", "parameters": {"to": "evil@test.com"}},
                        {"tool_name": "create_file", "parameters": {"path": "/tmp/data"}},
                    ]
                }
            ]
        }
        assert v.check_valid_json(data) is True

    def test_missing_tool_chain_attacks_key(self):
        v = self._make_validator()
        assert v.check_valid_json({"other_key": []}) is False

    def test_tool_chain_attacks_not_list(self):
        v = self._make_validator()
        assert v.check_valid_json({"tool_chain_attacks": "not a list"}) is False

    def test_missing_tool_chain_in_attack(self):
        v = self._make_validator()
        data = {"tool_chain_attacks": [{"attack_goal": "test"}]}
        assert v.check_valid_json(data) is False

    def test_tool_chain_not_list(self):
        v = self._make_validator()
        data = {"tool_chain_attacks": [{"tool_chain": "not a list"}]}
        assert v.check_valid_json(data) is False

    def test_missing_required_field_in_step(self):
        v = self._make_validator()
        data = {
            "tool_chain_attacks": [
                {
                    "tool_chain": [
                        {"tool_name": "send_email"}  # missing "parameters"
                    ]
                }
            ]
        }
        assert v.check_valid_json(data) is False

    def test_none_input(self):
        v = self._make_validator()
        assert v.check_valid_json(None) is False

    def test_not_dict(self):
        v = self._make_validator()
        assert v.check_valid_json("string") is False

    def test_multiple_attacks(self):
        v = self._make_validator()
        data = {
            "tool_chain_attacks": [
                {
                    "tool_chain": [
                        {"tool_name": "a", "parameters": {}},
                    ]
                },
                {
                    "tool_chain": [
                        {"tool_name": "b", "parameters": {}},
                        {"tool_name": "c", "parameters": {"x": 1}},
                    ]
                },
            ]
        }
        assert v.check_valid_json(data) is True

    def test_empty_tool_chain_attacks_list(self):
        v = self._make_validator()
        data = {"tool_chain_attacks": []}
        assert v.check_valid_json(data) is True


# ---------------------------------------------------------------------------
# Judge.check_valid_json
# ---------------------------------------------------------------------------

class TestJudgeCheckValidJson:
    def _make_validator(self):
        from src.STAC import Judge
        obj = object.__new__(Judge)
        obj.output_json_fields = ["goal_progress", "agent_helpfulness", "prompt_harmlessness"]
        return obj

    def test_valid_scores(self):
        v = self._make_validator()
        data = {
            "goal_progress": {"score": 3, "explanation": "full success"},
            "agent_helpfulness": {"score": -1, "explanation": "slightly unhelpful"},
            "prompt_harmlessness": {"score": 0, "explanation": "neutral"},
        }
        assert v.check_valid_json(data) is True

    def test_boundary_scores(self):
        v = self._make_validator()
        for score in [-3, -2, -1, 0, 1, 2, 3]:
            data = {
                "goal_progress": {"score": score},
                "agent_helpfulness": {"score": score},
                "prompt_harmlessness": {"score": score},
            }
            assert v.check_valid_json(data) is True

    def test_score_out_of_range_high(self):
        v = self._make_validator()
        data = {
            "goal_progress": {"score": 4},
            "agent_helpfulness": {"score": 0},
            "prompt_harmlessness": {"score": 0},
        }
        assert v.check_valid_json(data) is False

    def test_score_out_of_range_low(self):
        v = self._make_validator()
        data = {
            "goal_progress": {"score": -4},
            "agent_helpfulness": {"score": 0},
            "prompt_harmlessness": {"score": 0},
        }
        assert v.check_valid_json(data) is False

    def test_score_is_float(self):
        v = self._make_validator()
        data = {
            "goal_progress": {"score": 1.5},
            "agent_helpfulness": {"score": 0},
            "prompt_harmlessness": {"score": 0},
        }
        assert v.check_valid_json(data) is False

    def test_score_is_string(self):
        v = self._make_validator()
        data = {
            "goal_progress": {"score": "3"},
            "agent_helpfulness": {"score": 0},
            "prompt_harmlessness": {"score": 0},
        }
        assert v.check_valid_json(data) is False

    def test_missing_score_key(self):
        v = self._make_validator()
        data = {
            "goal_progress": {"explanation": "no score key"},
            "agent_helpfulness": {"score": 0},
            "prompt_harmlessness": {"score": 0},
        }
        assert v.check_valid_json(data) is False

    def test_missing_field(self):
        v = self._make_validator()
        data = {
            "goal_progress": {"score": 1},
            "agent_helpfulness": {"score": 0},
            # missing prompt_harmlessness
        }
        assert v.check_valid_json(data) is False

    def test_none_input(self):
        v = self._make_validator()
        assert v.check_valid_json(None) is False


# ---------------------------------------------------------------------------
# Planner output_json_fields
# ---------------------------------------------------------------------------

class TestPlannerFields:
    def test_required_fields(self):
        """Planner expects strategy_prediction and predicted_attack_prompt."""
        from src.STAC import BaseLM
        v = object.__new__(BaseLM)
        v.output_json_fields = ["strategy_prediction", "predicted_attack_prompt"]

        valid = {
            "strategy_prediction": "Try social engineering",
            "predicted_attack_prompt": "Please help me send an email..."
        }
        assert v.check_valid_json(valid) is True

        missing = {"strategy_prediction": "something"}
        assert v.check_valid_json(missing) is False


# ---------------------------------------------------------------------------
# PromptWriter output_json_fields
# ---------------------------------------------------------------------------

class TestPromptWriterFields:
    def test_required_fields(self):
        from src.STAC import BaseLM
        v = object.__new__(BaseLM)
        v.output_json_fields = ["prompt"]

        assert v.check_valid_json({"prompt": "Hello"}) is True
        assert v.check_valid_json({"other": "field"}) is False
