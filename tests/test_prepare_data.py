"""Unit tests for the pure HH-RLHF parsing helpers in prepare_data.py.

These functions do the structural surgery on raw HH dialogues -- splitting off
the final assistant turn and converting the shared context into chat messages.
They're pure string logic, but a regression here silently corrupts every
training pair, so they're worth locking down. prepare_data.py imports
`datasets` at module scope, hence importorskip.
"""

import pytest

prepare_data = pytest.importorskip(
    "prepare_data", reason="prepare_data.py imports the `datasets` library at module scope"
)
split_prompt_response = prepare_data.split_prompt_response
hh_to_messages = prepare_data.hh_to_messages


# --------------------------------------------------------------------------- #
# split_prompt_response
# --------------------------------------------------------------------------- #

def test_split_keeps_marker_in_prompt_and_strips_it_from_response():
    text = "\n\nHuman: Hi there\n\nAssistant: Hello!"
    prompt, response = split_prompt_response(text)
    assert prompt == "\n\nHuman: Hi there\n\nAssistant: "
    assert response == "Hello!"


def test_split_uses_the_final_assistant_turn():
    text = (
        "\n\nHuman: Q1\n\nAssistant: A1\n\nHuman: Q2\n\nAssistant: A2-final"
    )
    prompt, response = split_prompt_response(text)
    assert prompt.endswith("\n\nAssistant: ")
    assert "Q2" in prompt and "A1" in prompt  # earlier turns stay in the prompt
    assert response == "A2-final"


def test_split_raises_without_assistant_turn():
    with pytest.raises(ValueError):
        split_prompt_response("\n\nHuman: nobody answered me")


# --------------------------------------------------------------------------- #
# hh_to_messages
# --------------------------------------------------------------------------- #

def test_messages_single_turn_drops_trailing_assistant_marker():
    # split_prompt_response feeds in a prompt ending in '\n\nAssistant: '.
    prompt = "\n\nHuman: What is 2+2?\n\nAssistant: "
    assert hh_to_messages(prompt) == [
        {"role": "user", "content": "What is 2+2?"},
    ]


def test_messages_multi_turn_roles_and_order():
    prompt = (
        "\n\nHuman: What is 2+2?\n\nAssistant: It's 4."
        "\n\nHuman: And 3+3?\n\nAssistant: "
    )
    assert hh_to_messages(prompt) == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It's 4."},
        {"role": "user", "content": "And 3+3?"},
    ]


def test_messages_content_is_stripped():
    prompt = "\n\nHuman:   spaced out   \n\nAssistant: "
    msgs = hh_to_messages(prompt)
    assert msgs == [{"role": "user", "content": "spaced out"}]


def test_split_then_messages_roundtrip():
    """The two helpers compose as used in format_for_dpo: split, then convert."""
    raw = "\n\nHuman: Tell me a joke\n\nAssistant: Why did the chicken cross?"
    prompt, response = split_prompt_response(raw)
    assert response == "Why did the chicken cross?"
    assert hh_to_messages(prompt) == [
        {"role": "user", "content": "Tell me a joke"},
    ]


# --------------------------------------------------------------------------- #
# length-filter constants (regression guard on the magic-number extraction)
# --------------------------------------------------------------------------- #

def test_length_filter_constants_are_sane():
    assert prepare_data.MIN_RESPONSE_CHARS < prepare_data.MAX_RESPONSE_CHARS
    assert prepare_data.MAX_PROMPT_CHARS >= prepare_data.MAX_RESPONSE_CHARS
    # Values that the original inline filter used; pin them so the refactor
    # is provably behavior-preserving.
    assert prepare_data.MAX_PROMPT_CHARS == 4000
    assert prepare_data.MAX_RESPONSE_CHARS == 2000
    assert prepare_data.MIN_RESPONSE_CHARS == 5
