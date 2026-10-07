"""Per-user conversation memory so follow-up questions work (FR11: multiple users).
In-memory and thread-safe; for real deployment, swap for Redis or a database."""
import threading
import time


class SessionStore:
    def __init__(self, max_turns: int = 3, ttl_seconds: int = 1800):
        self.max_messages = max_turns * 2
        self.ttl = ttl_seconds
        self._data: dict[str, tuple[float, list[dict]]] = {}
        self._lock = threading.Lock()

    def get(self, sid: str) -> list[dict]:
        with self._lock:
            self._expire()
            return list(self._data.get(sid, (0, []))[1])

    def append(self, sid: str, question: str, answer: str) -> None:
        with self._lock:
            history = list(self._data.get(sid, (0, []))[1])
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer}]
            self._data[sid] = (time.time(), history[-self.max_messages:])

    def clear(self, sid: str) -> None:
        with self._lock:
            self._data.pop(sid, None)

    def _expire(self) -> None:
        now = time.time()
        for sid in [s for s, (t, _) in self._data.items() if now - t > self.ttl]:
            del self._data[sid]
