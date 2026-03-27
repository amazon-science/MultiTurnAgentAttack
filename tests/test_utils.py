"""Unit tests for src/utils.py

Run:  python -m pytest tests/test_utils.py -v
"""

from __future__ import annotations

import json
from typing import Annotated, Dict, List, Optional, Union

import pytest

from src.utils import (
    batchify,
    convert_message_between_APIs,
    gen_tool_call_id,
    get_failure_mode,
    get_json_type_as_string,
    get_schema_from_annotation,
    str2json,
)


# ---------------------------------------------------------------------------
# str2json
# ---------------------------------------------------------------------------

class TestStr2Json:
    def test_plain_json(self):
        assert str2json('{"key": "value"}') == {"key": "value"}

    def test_json_with_surrounding_text(self):
        result = str2json('Here is the output: {"a": 1, "b": 2} done.')
        assert result == {"a": 1, "b": 2}

    def test_json_code_block(self):
        text = '```json\n{"tool_name": "send_email", "parameters": {}}\n```'
        result = str2json(text)
        assert result["tool_name"] == "send_email"

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = str2json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_invalid_returns_empty(self):
        # json_repair returns empty string for non-JSON input
        assert str2json("no json here") == ""

    def test_empty_string(self):
        assert str2json("") == ""


# ---------------------------------------------------------------------------
# batchify
# ---------------------------------------------------------------------------

class TestBatchify:
    def test_even_split(self):
        batches = list(batchify([1, 2, 3, 4], 2))
        assert batches == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        batches = list(batchify([1, 2, 3, 4, 5], 2))
        assert batches == [[1, 2], [3, 4], [5]]

    def test_batch_larger_than_data(self):
        batches = list(batchify([1, 2], 10))
        assert batches == [[1, 2]]

    def test_empty_data(self):
        assert list(batchify([], 5)) == []

    def test_batch_size_one(self):
        batches = list(batchify([1, 2, 3], 1))
        assert batches == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# gen_tool_call_id
# ---------------------------------------------------------------------------

class TestGenToolCallId:
    def test_returns_string(self):
        assert isinstance(gen_tool_call_id(), str)

    def test_unique(self):
        ids = {gen_tool_call_id() for _ in range(100)}
        assert len(ids) == 100

    def test_hex_format(self):
        tid = gen_tool_call_id()
        assert len(tid) == 32
        int(tid, 16)  # should not raise


# ---------------------------------------------------------------------------
# get_failure_mode
# ---------------------------------------------------------------------------

class TestGetFailureMode:
    @pytest.mark.parametrize("fid", range(1, 11))
    def test_valid_ids(self, fid):
        result = get_failure_mode(fid)
        assert isinstance(result, str)
        assert "Definition:" in result

    def test_invalid_id_returns_none(self):
        assert get_failure_mode(0) is None
        assert get_failure_mode(11) is None


# ---------------------------------------------------------------------------
# get_json_type_as_string
# ---------------------------------------------------------------------------

class TestGetJsonTypeAsString:
    def test_basic_types(self):
        assert get_json_type_as_string(str) == "string"
        assert get_json_type_as_string(int) == "integer"
        assert get_json_type_as_string(float) == "number"
        assert get_json_type_as_string(bool) == "boolean"
        assert get_json_type_as_string(type(None)) == "null"

    def test_list(self):
        assert get_json_type_as_string(List[str]) == "array"

    def test_dict(self):
        assert get_json_type_as_string(Dict[str, int]) == "object"

    def test_optional(self):
        assert get_json_type_as_string(Optional[str]) == "string"

    def test_union_with_none(self):
        assert get_json_type_as_string(Union[int, None]) == "integer"


# ---------------------------------------------------------------------------
# get_schema_from_annotation
# ---------------------------------------------------------------------------

class TestGetSchemaFromAnnotation:
    def test_str(self):
        assert get_schema_from_annotation(str) == {"type": "string"}

    def test_int(self):
        assert get_schema_from_annotation(int) == {"type": "number"}

    def test_bool(self):
        assert get_schema_from_annotation(bool) == {"type": "boolean"}

    def test_list_of_str(self):
        schema = get_schema_from_annotation(List[str])
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_optional_int(self):
        schema = get_schema_from_annotation(Optional[int])
        assert schema["type"] == "number"

    def test_annotated_with_description(self):
        schema = get_schema_from_annotation(Annotated[str, "A description"])
        assert schema["type"] == "string"
        assert schema["description"] == "A description"

    def test_annotated_nested_list(self):
        schema = get_schema_from_annotation(Annotated[List[int], "list of ints"])
        assert schema["type"] == "array"
        assert schema["description"] == "list of ints"


# ---------------------------------------------------------------------------
# convert_message_between_APIs
# ---------------------------------------------------------------------------

class TestConvertMessageBetweenAPIs:
    """Tests for cross-API message format conversion."""

    # -- User messages --

    def test_user_str_to_bedrock(self):
        msg = {"role": "user", "content": "hello"}
        result = convert_message_between_APIs(msg, "claude-3")
        assert result["role"] == "user"
        assert result["content"] == [{"text": "hello"}]

    def test_user_str_to_gpt(self):
        msg = {"role": "user", "content": "hello"}
        result = convert_message_between_APIs(msg, "gpt-4")
        assert result == msg

    def test_user_bedrock_to_gpt(self):
        msg = {"role": "user", "content": [{"text": "hello"}]}
        result = convert_message_between_APIs(msg, "gpt-4")
        assert result["content"] == "hello"

    def test_user_bedrock_to_bedrock(self):
        # Note: convert_message_between_APIs wraps list content again for claude
        # This is a known quirk — the function re-wraps content in [{"text": ...}]
        msg = {"role": "user", "content": [{"text": "hello"}]}
        result = convert_message_between_APIs(msg, "claude-3")
        assert result["role"] == "user"
        assert result["content"] == [{"text": [{"text": "hello"}]}]

    # -- Assistant messages (GPT format with tool_calls) --

    def test_assistant_gpt_tool_call_to_bedrock(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": "tc_1",
                "function": {
                    "name": "send_email",
                    "arguments": '{"to": "alice@test.com"}'
                }
            }]
        }
        result = convert_message_between_APIs(msg, "claude-3")
        assert result["role"] == "assistant"
        tool_use = result["content"][0]["toolUse"]
        assert tool_use["name"] == "send_email"
        assert tool_use["toolUseId"] == "tc_1"
        assert tool_use["input"] == {"to": "alice@test.com"}

    def test_assistant_gpt_tool_call_to_gpt(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": "tc_1",
                "function": {
                    "name": "send_email",
                    "arguments": '{"to": "alice@test.com"}'
                }
            }]
        }
        result = convert_message_between_APIs(msg, "gpt-4.1")
        assert result == msg

    def test_assistant_gpt_tool_call_to_vllm(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": "tc_1",
                "function": {
                    "name": "send_email",
                    "arguments": '{"to": "alice@test.com"}'
                }
            }]
        }
        result = convert_message_between_APIs(msg, "Qwen/Qwen3-32B")
        assert "<tool_call>" in result["content"]
        assert "send_email" in result["content"]

    # -- Assistant messages (plain text) --

    def test_assistant_text_to_bedrock(self):
        msg = {"role": "assistant", "content": "Sure, I can help."}
        result = convert_message_between_APIs(msg, "claude-3")
        assert result["content"] == [{"text": "Sure, I can help."}]

    def test_assistant_text_to_gpt(self):
        msg = {"role": "assistant", "content": "Sure, I can help."}
        result = convert_message_between_APIs(msg, "gpt-4")
        assert result == msg

    # -- Tool result messages (GPT format) --

    def test_tool_gpt_to_bedrock(self):
        msg = {
            "role": "tool",
            "tool_call_id": "tc_1",
            "name": "send_email",
            "content": "Email sent successfully"
        }
        result = convert_message_between_APIs(msg, "claude-3")
        assert result["role"] == "tool"
        tool_result = result["content"][0]["toolResult"]
        assert tool_result["toolUseId"] == "tc_1"
        assert tool_result["content"][0]["text"] == "Email sent successfully"

    def test_tool_gpt_to_gpt(self):
        msg = {
            "role": "tool",
            "tool_call_id": "tc_1",
            "name": "send_email",
            "content": "Email sent successfully"
        }
        result = convert_message_between_APIs(msg, "gpt-4")
        assert result == msg

    # -- Tool result messages (Bedrock format) --

    def test_tool_bedrock_to_gpt(self):
        msg = {
            "role": "tool",
            "content": [{
                "toolResult": {
                    "toolUseId": "tc_1",
                    "content": [{"text": "Done"}]
                }
            }]
        }
        result = convert_message_between_APIs(msg, "gpt-4")
        assert result["tool_call_id"] == "tc_1"
        assert result["content"] == "Done"

    def test_tool_bedrock_to_bedrock(self):
        msg = {
            "role": "tool",
            "content": [{
                "toolResult": {
                    "toolUseId": "tc_1",
                    "content": [{"text": "Done"}]
                }
            }]
        }
        result = convert_message_between_APIs(msg, "claude-3")
        assert result == msg

    # -- Assistant messages (Bedrock format with toolUse) --

    def test_assistant_bedrock_to_gpt(self):
        msg = {
            "role": "assistant",
            "content": [{
                "toolUse": {
                    "toolUseId": "tc_1",
                    "name": "get_balance",
                    "input": {"account": "123"}
                }
            }]
        }
        result = convert_message_between_APIs(msg, "gpt-4.1")
        assert "tool_calls" in result
        tc = result["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["function"]["name"] == "get_balance"
        assert json.loads(tc["function"]["arguments"]) == {"account": "123"}

    def test_assistant_bedrock_to_bedrock(self):
        msg = {
            "role": "assistant",
            "content": [{
                "toolUse": {
                    "toolUseId": "tc_1",
                    "name": "get_balance",
                    "input": {"account": "123"}
                }
            }]
        }
        result = convert_message_between_APIs(msg, "claude-3")
        assert result == msg
