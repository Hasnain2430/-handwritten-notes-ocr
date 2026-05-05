import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AgentPlanner:
    def __init__(self):
        self.api_key = os.getenv("AGENT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("AGENT_LLM_BASE_URL")
        self.model = os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")

    def plan_page(
        self,
        user_goal: str,
        image_quality: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.api_key:
            try:
                return self._llm_plan(user_goal, image_quality, memory_hits)
            except Exception as exc:
                logger.warning("Agent LLM planner failed, using heuristic planner: %s", exc)
        return self._heuristic_plan(image_quality, memory_hits)

    def _llm_plan(
        self,
        user_goal: str,
        image_quality: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        from openai import OpenAI

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        prompt = {
            "user_goal": user_goal,
            "image_quality": image_quality,
            "relevant_memory": memory_hits[:3],
            "allowed_actions": [
                "preprocess_image",
                "detect_layout",
                "run_ocr",
                "self_critique",
                "request_human_review",
            ],
        }
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR workflow planner. Return only compact JSON with keys: actions, rationale, human_review_threshold.",
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def _heuristic_plan(self, image_quality: Dict[str, Any], memory_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        actions = ["preprocess_image", "detect_layout", "run_ocr", "self_critique"]
        recommendations = image_quality.get("recommendations", [])
        if recommendations or memory_hits:
            actions.append("request_human_review_if_uncertain")
        return {
            "actions": actions,
            "rationale": "Heuristic plan based on image quality and available long-term memory.",
            "human_review_threshold": "review if blur is low, contrast is low, OCR is empty, or symbol noise is high",
        }
