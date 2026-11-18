"""Answer faithfulness: LLM judge over decomposed claims, with lexical fallback."""
from __future__ import annotations

import json
import re

from app.config import Settings

from .metrics import lexical_faithfulness, split_claims

JUDGE_SYSTEM = (
    "You are a strict fact-checker. Given CONTEXT and a list of CLAIMS, decide for each claim "
    "whether it is fully supported by the CONTEXT alone (no outside knowledge). "
    'Return ONLY a JSON list of booleans, one per claim, e.g. [true,false].'
)


class FaithfulnessScorer:
    def __init__(self, s: Settings):
        self.s = s
        self.client = None
        if s.anthropic_api_key:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=s.anthropic_api_key)

    @property
    def method(self) -> str:
        return "llm_judge" if self.client else "lexical"

    async def score(self, answer: str, contexts: list[str]) -> float:
        if "i don't know" in answer.lower():
            return 1.0
        if not self.client:
            return lexical_faithfulness(answer, contexts)
        claims = split_claims(answer)
        if not claims:
            return 0.0
        prompt = (
            "CONTEXT:\n" + "\n\n".join(contexts) + "\n\nCLAIMS:\n"
            + "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        )
        msg = await self.client.messages.create(
            model=self.s.llm_model, max_tokens=200, temperature=0,
            system=JUDGE_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        m = re.search(r"\[.*?\]", text, re.S)
        try:
            verdicts = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            verdicts = []
        if len(verdicts) != len(claims):
            return lexical_faithfulness(answer, contexts)  # malformed judge output -> fallback
        return sum(bool(v) for v in verdicts) / len(verdicts)
