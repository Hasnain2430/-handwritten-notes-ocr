import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.agent.memory import AgentMemory
from app.agent.planner import AgentPlanner
from app.agent.state import AgentPageResult, AgentState
from app.agent.tools import (
    assess_image_quality,
    critique_structured_output,
    parse_structured_text,
    preprocess_with_policy,
    run_layout_detection,
    run_qwen_ocr,
)
from app.services.docx_generator import DOCXGenerator

logger = logging.getLogger(__name__)


class AgenticOCRAgent:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.memory = AgentMemory(base_dir)
        self.planner = AgentPlanner()

    async def run(
        self,
        image_paths: List[str],
        uploads_dir: Path,
        outputs_dir: Path,
        user_goal: str,
        session_id: str | None = None,
    ) -> AgentState:
        session_id = session_id or str(uuid.uuid4())
        state = AgentState(
            session_id=session_id,
            user_goal=user_goal,
            image_paths=image_paths,
            uploads_dir=uploads_dir,
            outputs_dir=outputs_dir,
        )
        self._log(state, "observe", "session_started", {
            "image_count": len(image_paths),
            "user_goal": user_goal,
        })

        state.memory_hits = self.memory.retrieve_relevant_profiles(user_goal)
        self._log(state, "interpret", "retrieved_long_term_memory", {
            "memory_hits": len(state.memory_hits),
        })

        all_structured_json: List[Dict[str, Any]] = []
        for index, image_path in enumerate(image_paths):
            state.current_page = index
            page_result = await self._process_page(state, image_path, index)
            state.page_results.append(page_result)

            if index > 0:
                all_structured_json.append({"type": "page_break"})
            all_structured_json.extend(page_result.structured_json)

            if page_result.critique.get("human_review_required"):
                state.human_review_required = True
                state.human_review_questions.append(
                    f"Page {index + 1} may need review because {page_result.critique.get('summary', 'OCR uncertainty was detected')}."
                )

        if not all_structured_json:
            all_structured_json = [{
                "type": "paragraph",
                "text": "[Agentic OCR pipeline executed but no readable text was extracted]",
                "source": "agent_empty_fallback",
            }]

        output_path = outputs_dir / f"agentic_{session_id}_converted.docx"
        DOCXGenerator().generate_document(
            structured_json=all_structured_json,
            diagram_dir=None,
            output_path=output_path,
        )
        state.final_document_path = str(output_path)
        state.confidence_report = self._build_confidence_report(state)

        self.memory.remember_document_profile({
            "user_goal": user_goal,
            "image_count": len(image_paths),
            "human_review_required": state.human_review_required,
            "confidence_report": state.confidence_report,
        })
        self.memory.save_session_event(session_id, {
            "type": "agent_run_completed",
            "output_path": str(output_path),
            "human_review_required": state.human_review_required,
            "confidence_report": state.confidence_report,
        })
        self._log(state, "learn", "document_profile_saved", {
            "output_path": str(output_path),
        })
        return state

    async def _process_page(self, state: AgentState, image_path: str, page_index: int) -> AgentPageResult:
        page = AgentPageResult(page_index=page_index, original_path=image_path)

        self._log(state, "observe", "assessing_image_quality", {"page": page_index + 1})
        page.quality = assess_image_quality(image_path)
        page.actions.append({"action": "assess_image_quality", "result": page.quality})

        plan = self.planner.plan_page(state.user_goal, page.quality, state.memory_hits)
        page.actions.append({"action": "plan_page", "result": plan})
        self._log(state, "decide", "planner_selected_actions", {
            "page": page_index + 1,
            "plan": plan,
        })

        self._log(state, "decide", "selecting_preprocessing_policy", {
            "page": page_index + 1,
            "recommendations": page.quality.get("recommendations", []),
        })
        processed_path, preprocess_actions = preprocess_with_policy(image_path, page.quality)
        page.processed_path = processed_path
        page.actions.extend(preprocess_actions)

        self._log(state, "act", "detecting_layout", {"page": page_index + 1})
        regions, layout_action = run_layout_detection(processed_path)
        page.layout_regions = regions
        page.actions.append(layout_action)

        self._log(state, "act", "running_ocr_tool", {"page": page_index + 1})
        try:
            text, diagram_regions, ocr_action = run_qwen_ocr(processed_path)
            page.actions.append(ocr_action)
            page.actions.append({
                "action": "diagram_region_detection",
                "decision": "recorded",
                "regions": diagram_regions,
            })
            page.structured_json = parse_structured_text(text)
        except Exception as exc:
            logger.exception("Agent OCR failed on page %s", page_index + 1)
            page.actions.append({
                "action": "qwen_vl_ocr",
                "decision": "failed",
                "reason": str(exc),
            })
            page.structured_json = [{
                "type": "paragraph",
                "text": f"[OCR ERROR: {str(exc)}]",
                "source": "agent_ocr_error",
            }]

        self._log(state, "interpret", "self_critiquing_page_output", {"page": page_index + 1})
        page.critique = critique_structured_output(page.structured_json, page.quality)
        page.actions.append({"action": "self_critique", "result": page.critique})

        self.memory.save_session_event(state.session_id, {
            "type": "page_processed",
            "page": page_index + 1,
            "quality": page.quality,
            "critique": page.critique,
            "actions": page.actions,
        })
        return page

    def _build_confidence_report(self, state: AgentState) -> Dict[str, Any]:
        pages = []
        review_count = 0
        for page in state.page_results:
            critique = page.critique
            if critique.get("human_review_required"):
                review_count += 1
            pages.append({
                "page": page.page_index + 1,
                "quality": page.quality,
                "elements": critique.get("total_elements", len(page.structured_json)),
                "characters": critique.get("total_characters", 0),
                "human_review_required": critique.get("human_review_required", False),
                "suspicious_items": critique.get("suspicious_items", []),
            })
        return {
            "pages": pages,
            "total_pages": len(state.page_results),
            "pages_requiring_review": review_count,
            "overall_status": "human_review_recommended" if review_count else "agent_completed",
        }

    def _log(self, state: AgentState, phase: str, message: str, data: Dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "message": message,
            "data": data,
        }
        state.audit_log.append(event)
        logger.info("Agent phase=%s message=%s data=%s", phase, message, data)
