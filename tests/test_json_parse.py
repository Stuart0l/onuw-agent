import json

import pytest

from onuw.utils.json_parse import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_plain_array():
    assert extract_json('[1, 2, 3]') == [1, 2, 3]


def test_strips_markdown_fence_with_language():
    raw = '```json\n{"action": "rob", "target": "p2"}\n```'
    assert extract_json(raw) == {"action": "rob", "target": "p2"}


def test_strips_markdown_fence_without_language():
    raw = '```\n{"vote": "p1"}\n```'
    assert extract_json(raw) == {"vote": "p1"}


def test_handles_leading_prose():
    raw = 'Sure! Here is my answer:\n{"speech": "I think p3 is a wolf."}'
    assert extract_json(raw) == {"speech": "I think p3 is a wolf."}


def test_handles_trailing_prose():
    raw = '{"speech": "hi"}\nLet me know if you need more.'
    assert extract_json(raw) == {"speech": "hi"}


def test_nested_objects_balanced_correctly():
    raw = 'pre {"a": {"b": 1, "c": [1, 2]}} post'
    assert extract_json(raw) == {"a": {"b": 1, "c": [1, 2]}}


def test_quoted_braces_inside_string_dont_confuse_matcher():
    raw = '{"speech": "I like }{} characters"}'
    assert extract_json(raw) == {"speech": "I like }{} characters"}


def test_escaped_quotes_inside_string():
    raw = r'{"speech": "He said \"hi\" to me"}'
    assert extract_json(raw) == {"speech": 'He said "hi" to me'}


def test_raises_on_pure_prose():
    with pytest.raises(json.JSONDecodeError):
        extract_json("just prose, no JSON here at all")


def test_raises_on_empty():
    with pytest.raises(json.JSONDecodeError):
        extract_json("")


def test_raises_on_unbalanced_object():
    with pytest.raises(json.JSONDecodeError):
        extract_json('{"a": 1')