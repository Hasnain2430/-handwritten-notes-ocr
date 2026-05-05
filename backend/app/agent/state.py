from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentPageResult:
    page_index: int
    original_path: str
    processed_path: Optional[str] = None
    quality: Dict[str, Any] = field(default_factory=dict)
    layout_regions: List[Dict[str, Any]] = field(default_factory=list)
    structured_json: List[Dict[str, Any]] = field(default_factory=list)
    critique: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentState:
    session_id: str
    user_goal: str
    image_paths: List[str]
    uploads_dir: Path
    outputs_dir: Path
    current_page: int = 0
    page_results: List[AgentPageResult] = field(default_factory=list)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    human_review_required: bool = False
    human_review_questions: List[str] = field(default_factory=list)
    final_document_path: Optional[str] = None
    confidence_report: Dict[str, Any] = field(default_factory=dict)
