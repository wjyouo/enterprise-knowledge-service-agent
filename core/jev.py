"""
Optional Jev decision client for typed routing decisions.

Modified for the IT support ticket-agent adaptation. Jev is used only for
closed decisions that code can branch on directly: query route selection,
ticket action selection, urgency scoring, and human-review gates. It never
generates user-facing text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


ROUTE_OPTIONS = {
    "general_knowledge": "No internal documents or tools are needed.",
    "retrieval_needed": "Internal knowledge base documents are needed.",
    "tool_needed": "A business tool or API call is needed.",
    "both_needed": "Both internal documents and a business tool are needed.",
}

TICKET_ACTION_OPTIONS = {
    "answer_from_knowledge": "Answer from IT knowledge or policy content.",
    "create_ticket": "Open a new support ticket.",
    "query_tickets": "Search tickets with filters such as owner, date, status, or priority.",
    "list_tickets": "List tickets without filters.",
    "escalate_to_human": "A human should review before any automated action.",
}


@dataclass(frozen=True)
class JevAnswer:
    name: str
    raw: Any
    choice: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    score: Optional[float] = None
    noul: Optional[float] = None


@dataclass(frozen=True)
class JevRouteDecision:
    query_type: str
    need_retrieval: bool
    need_tools: bool
    confidence: Optional[float]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class JevSupportTriage:
    ticket_action: str
    action_confidence: Optional[float]
    urgency_score: Optional[float]
    needs_human_review: Optional[float]
    raw: Dict[str, Any]


def jev_api_key() -> str:
    return settings.jev_api_key or settings.typesafe_api_key


def jev_is_configured() -> bool:
    return bool(jev_api_key())


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _float_map(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, Mapping):
        return None

    out: Dict[str, float] = {}
    for key, raw_score in value.items():
        score = _as_float(raw_score)
        if score is not None:
            out[str(key)] = score
    return out or None


def _choice_from_mapping(data: Mapping[str, Any]) -> Optional[str]:
    for field in ("choice", "label", "value", "answer", "selected"):
        value = data.get(field)
        if isinstance(value, str):
            return value

    weighted = _float_map(data.get("probabilities")) or _float_map(data.get("scores"))
    if weighted:
        return max(weighted, key=weighted.get)
    return None


def _confidence_for_choice(data: Mapping[str, Any], choice: Optional[str]) -> Optional[float]:
    for field in ("confidence", "probability", "p"):
        value = _as_float(data.get(field))
        if value is not None:
            return value

    weighted = _float_map(data.get("probabilities")) or _float_map(data.get("scores"))
    if weighted and choice in weighted:
        return weighted[choice]
    return None


def normalize_jev_answer(name: str, raw: Any) -> JevAnswer:
    if isinstance(raw, str):
        return JevAnswer(name=name, raw=raw, choice=raw)

    scalar = _as_float(raw)
    if scalar is not None:
        return JevAnswer(name=name, raw=raw, score=scalar, noul=scalar)

    if not isinstance(raw, Mapping):
        return JevAnswer(name=name, raw=raw)

    choice = _choice_from_mapping(raw)
    confidence = _confidence_for_choice(raw, choice)
    probabilities = _float_map(raw.get("probabilities")) or _float_map(raw.get("scores"))
    score = None
    for field in ("score", "rating"):
        score = _as_float(raw.get(field))
        if score is not None:
            break
    noul = None
    for field in ("noul", "probability", "p_true", "answer", "value"):
        noul = _as_float(raw.get(field))
        if noul is not None:
            break

    return JevAnswer(
        name=name,
        raw=dict(raw),
        choice=choice,
        confidence=confidence,
        probabilities=probabilities,
        score=score,
        noul=noul,
    )


def _answers_from_response(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("answers", "results", "data", "output"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def route_flags_from_choice(choice: str) -> Dict[str, bool]:
    if choice == "both_needed":
        return {"need_retrieval": True, "need_tools": True}
    if choice == "retrieval_needed":
        return {"need_retrieval": True, "need_tools": False}
    if choice == "tool_needed":
        return {"need_retrieval": False, "need_tools": True}
    return {"need_retrieval": False, "need_tools": False}


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in history[-6:]]
    return "\n".join(lines)


class JevDecisionClient:
    def __init__(self) -> None:
        self._api_key = jev_api_key()
        self._api_url = settings.jev_api_url
        self._model = settings.jev_model

    async def decide(self, state: str, questions: Mapping[str, Any]) -> Optional[Dict[str, JevAnswer]]:
        if not self._api_key:
            return None

        payload = {
            "model": self._model,
            "state": state,
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=settings.jev_timeout_seconds) as client:
                response = await client.post(self._api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception:
            logger.exception("Jev decision request failed; falling back to existing router")
            return None

        if not isinstance(data, Mapping):
            logger.warning("Jev returned a non-object payload; falling back to existing router")
            return None

        answers = _answers_from_response(data)
        return {
            name: normalize_jev_answer(name, answer)
            for name, answer in answers.items()
        }


async def classify_query_with_jev(
    question: str, chat_history: List[Dict[str, str]]
) -> Optional[JevRouteDecision]:
    state = (
        "Enterprise assistant routing task.\n"
        f"Conversation history:\n{_format_history(chat_history)}\n\n"
        f"Current user question: {question}"
    )
    questions = {
        "query_route": {
            "type": "choice",
            "options": list(ROUTE_OPTIONS),
            "criteria": ROUTE_OPTIONS,
        }
    }

    answers = await JevDecisionClient().decide(state, questions)
    if not answers:
        return None

    answer = answers.get("query_route")
    if not answer or not answer.choice or answer.choice not in ROUTE_OPTIONS:
        return None
    if answer.confidence is not None and answer.confidence < settings.jev_min_confidence:
        return None

    flags = route_flags_from_choice(answer.choice)
    return JevRouteDecision(
        query_type=answer.choice,
        need_retrieval=flags["need_retrieval"],
        need_tools=flags["need_tools"],
        confidence=answer.confidence,
        raw={"query_route": answer.raw},
    )


async def triage_support_request_with_jev(query: str) -> Optional[JevSupportTriage]:
    state = f"Internal IT support request:\n{query}"
    questions = {
        "ticket_action": {
            "type": "choice",
            "options": list(TICKET_ACTION_OPTIONS),
            "criteria": TICKET_ACTION_OPTIONS,
        },
        "urgency": {
            "type": "score",
            "rubric": [
                "0.0 routine information request",
                "0.25 minor inconvenience",
                "0.5 user is blocked",
                "0.75 team or customer impact",
                "1.0 security incident, data loss, or production outage",
            ],
        },
        "needs_human_review": {
            "type": "noul",
            "statement": (
                "A human support engineer should review before the system creates "
                "or changes any ticket."
            ),
        },
    }

    answers = await JevDecisionClient().decide(state, questions)
    if not answers:
        return None

    action = answers.get("ticket_action")
    if not action or not action.choice or action.choice not in TICKET_ACTION_OPTIONS:
        return None
    if action.confidence is not None and action.confidence < settings.jev_min_confidence:
        return None

    urgency = answers.get("urgency")
    needs_human = answers.get("needs_human_review")
    return JevSupportTriage(
        ticket_action=action.choice,
        action_confidence=action.confidence,
        urgency_score=urgency.score if urgency else None,
        needs_human_review=needs_human.noul if needs_human else None,
        raw={name: answer.raw for name, answer in answers.items()},
    )
