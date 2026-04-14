"""Intent classification using Mistral 7B via Ollama.

Intents: search_product, add_to_cart, remove_from_cart, set_constraint,
         checkout, get_recommendation, chit_chat
"""

from __future__ import annotations

import logging
from typing import Tuple

from src.llm import ollama_client as llm
from src.nlu import demo_parser
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

INTENTS = [
    "search_product",
    "add_to_cart",
    "remove_from_cart",
    "set_constraint",
    "checkout",
    "get_recommendation",
    "chit_chat",
]

_SYSTEM_PROMPT = """\
You are an intent classifier for a grocery shopping assistant.
Classify the user's message into exactly ONE of these intents:
- search_product: user wants to find/search for grocery items
- add_to_cart: user wants to add a specific item to their cart
- remove_from_cart: user wants to remove an item from their cart
- set_constraint: user is setting a constraint like budget, dietary restriction, or quantity
- checkout: user wants to proceed to checkout / complete the order
- get_recommendation: user wants suggestions or recommendations
- chit_chat: greeting, small talk, or off-topic message

Return ONLY a JSON object with two fields:
{
  "intent": "<one of the intent names above>",
  "confidence": <float 0.0 to 1.0>
}
"""

_FEW_SHOT = [
    {"role": "user", "content": "I need 2 gallons of organic whole milk under $6"},
    {"role": "assistant", "content": '{"intent": "search_product", "confidence": 0.97}'},
    {"role": "user", "content": "Add the first one to my cart"},
    {"role": "assistant", "content": '{"intent": "add_to_cart", "confidence": 0.95}'},
    {"role": "user", "content": "Remove the eggs from my cart"},
    {"role": "assistant", "content": '{"intent": "remove_from_cart", "confidence": 0.96}'},
    {"role": "user", "content": "My budget is $50 and I'm gluten-free"},
    {"role": "assistant", "content": '{"intent": "set_constraint", "confidence": 0.94}'},
    {"role": "user", "content": "OK let's check out"},
    {"role": "assistant", "content": '{"intent": "checkout", "confidence": 0.99}'},
    {"role": "user", "content": "What snacks would go well with this?"},
    {"role": "assistant", "content": '{"intent": "get_recommendation", "confidence": 0.91}'},
    {"role": "user", "content": "Hey how's it going"},
    {"role": "assistant", "content": '{"intent": "chit_chat", "confidence": 0.98}'},
]


def classify(user_message: str, conversation_history: str = "") -> Tuple[str, float]:
    """Classify intent. Returns (intent_name, confidence)."""
    app_cfg = get_settings().get("app", {})
    if app_cfg.get("demo_mode", False):
        return demo_parser.classify_intent(user_message, conversation_history)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(_FEW_SHOT)

    # Include recent conversation context if available
    if conversation_history.strip():
        messages.append({
            "role": "user",
            "content": f"[Conversation so far]\n{conversation_history}\n\n[Current message]\n{user_message}",
        })
    else:
        messages.append({"role": "user", "content": user_message})

    try:
        result = llm.generate_json(
            prompt="\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages),
            role="nlu",
        )
        intent = result.get("intent", "chit_chat")
        confidence = float(result.get("confidence", 0.5))

        # Validate intent is known
        if intent not in INTENTS:
            logger.warning("Unknown intent '%s' from LLM, defaulting to chit_chat", intent)
            intent = "chit_chat"
            confidence = 0.5

        return intent, confidence

    except (ValueError, KeyError) as exc:
        logger.error("Intent classification failed: %s", exc)
        return demo_parser.classify_intent(user_message, conversation_history)
