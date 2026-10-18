"""Plain-language machine summaries.

Default: a deterministic template. Optional: any OpenAI-compatible endpoint (vLLM / SGLang on an
AMD Instinct GPU, Ollama, ...) via TELLTALE_LLM_BASE_URL. The LLM only *rephrases* facts computed
by the models — it never produces risk scores, and the template is always the fallback.
"""

import json
import logging
from typing import Any

import httpx

from ..core.config import Settings

log = logging.getLogger("telltale.narrator")

SYSTEM_PROMPT = (
    "You are a maintenance assistant for a factory floor. Write a short briefing (3-5 sentences) for a "
    "maintenance technician using ONLY the JSON facts provided. Do not invent numbers, causes or parts. "
    "Lead with the action, then the evidence. If risk is low, say so briefly."
)


class Narrator:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def llm_enabled(self) -> bool:
        return bool(self.settings.llm_base_url)

    async def summarize(self, facts: dict[str, Any]) -> tuple[str, str]:
        template = self.template(facts)
        if not self.llm_enabled:
            return template, "template"
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "temperature": 0.2,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(facts, default=str)},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
                r = await client.post(url, json=payload, headers={"Authorization": f"Bearer {self.settings.llm_api_key}"})
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"].strip()
                return text, "llm"
        except Exception as exc:
            log.warning("LLM summary failed (%s); using template", exc)
            return template, "template"

    @staticmethod
    def template(facts: dict[str, Any]) -> str:
        name = facts.get("machine", "This machine")
        parts = []
        sensor = facts.get("sensor")
        if sensor:
            p, level = sensor["failure_probability"], sensor["risk_level"]
            parts.append(f"{name} has a {level} estimated failure risk ({p:.0%}).")
            if level != "low" and sensor.get("top_mode"):
                parts.append(f"The most likely mechanism is {sensor['top_mode']}.")
            if level != "low" and sensor.get("drivers"):
                parts.append("Main drivers: " + ", ".join(sensor["drivers"]) + ".")
        else:
            parts.append(f"No sensor risk estimate is available for {name}.")
        sound = facts.get("acoustic")
        if sound:
            parts.append(f"Its last sound check was {sound['status']} (score {sound['severity']:.1f}x threshold).")
        actions = facts.get("actions") or []
        if actions:
            parts.append("Recommended next step: " + actions[0] + ".")
        return " ".join(parts)
