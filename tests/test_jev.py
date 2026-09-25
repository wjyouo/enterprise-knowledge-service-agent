"""Unit tests for Jev decision parsing. These do not call the external API."""
from core.jev import normalize_jev_answer, route_flags_from_choice


def test_normalize_choice_answer_with_probabilities():
    answer = normalize_jev_answer(
        "query_route",
        {
            "probabilities": {
                "general_knowledge": 0.05,
                "retrieval_needed": 0.1,
                "tool_needed": 0.15,
                "both_needed": 0.7,
            }
        },
    )

    assert answer.choice == "both_needed"
    assert answer.confidence == 0.7


def test_normalize_noul_probability():
    answer = normalize_jev_answer("needs_human_review", {"noul": 0.81})

    assert answer.noul == 0.81


def test_route_flags_for_both_path():
    assert route_flags_from_choice("both_needed") == {
        "need_retrieval": True,
        "need_tools": True,
    }


def test_route_flags_for_direct_path():
    assert route_flags_from_choice("general_knowledge") == {
        "need_retrieval": False,
        "need_tools": False,
    }
