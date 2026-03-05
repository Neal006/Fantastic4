"""In-memory multi-turn conversation manager with session support."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class ConversationManager:
    def __init__(self, max_turns: int = 20):
        self._sessions: Dict[str, List[Dict]] = defaultdict(list)
        self._max_turns = max_turns

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if session_id and session_id in self._sessions:
            return session_id
        new_id = session_id or str(uuid.uuid4())
        self._sessions[new_id] = []
        return new_id

    def add_message(self, session_id: str, role: str, content: str):
        self._sessions[session_id].append(
            {"role": role, "content": content, "ts": datetime.utcnow().isoformat()}
        )
        # Keep only the most recent turns
        if len(self._sessions[session_id]) > self._max_turns:
            self._sessions[session_id] = self._sessions[session_id][
                -self._max_turns :
            ]

    def get_history(self, session_id: str) -> List[Dict]:
        """Return history in OpenAI messages format (role + content only)."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._sessions.get(session_id, [])
        ]

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())
