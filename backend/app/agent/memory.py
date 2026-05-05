import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AgentMemory:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.memory_dir = self.base_dir / "agent_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.long_term_file = self.memory_dir / "long_term_memory.json"

    def load_session(self, session_id: str) -> Dict[str, Any]:
        session_file = self.memory_dir / f"session_{session_id}.json"
        if not session_file.exists():
            return {"session_id": session_id, "events": [], "preferences": {}}
        try:
            return json.loads(session_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read agent session memory %s: %s", session_id, exc)
            return {"session_id": session_id, "events": [], "preferences": {}}

    def save_session_event(self, session_id: str, event: Dict[str, Any]) -> None:
        session = self.load_session(session_id)
        session.setdefault("events", []).append({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            **event,
        })
        session_file = self.memory_dir / f"session_{session_id}.json"
        session_file.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_long_term(self) -> List[Dict[str, Any]]:
        if not self.long_term_file.exists():
            return []
        try:
            data = json.loads(self.long_term_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Failed to read long-term agent memory: %s", exc)
            return []

    def remember_document_profile(self, profile: Dict[str, Any]) -> None:
        records = self.load_long_term()
        records.append({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            **profile,
        })
        self.long_term_file.write_text(json.dumps(records[-250:], indent=2, ensure_ascii=False), encoding="utf-8")

    def retrieve_relevant_profiles(self, user_goal: str, limit: int = 5) -> List[Dict[str, Any]]:
        records = self.load_long_term()
        terms = {term.lower() for term in user_goal.split() if len(term) > 3}
        scored = []
        for record in records:
            haystack = " ".join(str(value) for value in record.values()).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]
