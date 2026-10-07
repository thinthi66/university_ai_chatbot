"""Contact / escalation handling (Person 3, Tasks 4 and 5)."""
import json
import re
from pathlib import Path

# Personal problems a staff member must handle: the bot gives a contact instead of trying to fix it.
_PERSONAL = re.compile(r"\b(my|me|i|i'm|im|i've)\b")
_PROBLEM = re.compile(
    r"\b(problem|issue|complain\w*|appeal|wrong|mistake|error|stuck|locked out|not working|"
    r"can'?t access|cannot access|unable to access|haven'?t received|didn'?t receive)\b")
_WHO_CONTACT = re.compile(r"\bwho (should|do|can|must) i (contact|speak to|talk to|email|call|see)\b")


def needs_human(question: str) -> bool:
    t = question.lower()
    return bool(_WHO_CONTACT.search(t) or (_PERSONAL.search(t) and _PROBLEM.search(t)))


class ContactDirectory:
    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.default = data.get("default", "general")
        self.departments = data["departments"]
        self.category_map = data.get("category_map", {})
        self._patterns = {
            key: [re.compile(r"\b" + re.escape(kw.lower())) for kw in d.get("keywords", [])]
            for key, d in self.departments.items()
        }

    def _from_category(self, category: str | None) -> str | None:
        """'current_students/module_registration' -> exact match first, then its folder."""
        if not category:
            return None
        for key in (category, category.split("/")[0]):
            if key in self.category_map:
                return self.category_map[key]
        return None

    def route(self, text: str, category_hint: str | None = None) -> dict:
        """Pick the department whose keywords best match the question; otherwise use the
        knowledge-base category of the best retrieved chunk; otherwise the main switchboard."""
        t = text.lower()
        best, best_hits = None, 0   # ties go to the department listed first (most specific)
        for key, pats in self._patterns.items():
            hits = sum(1 for p in pats if p.search(t))
            if hits > best_hits:
                best, best_hits = key, hits
        if best is None:
            best = self._from_category(category_hint) or self.default
        d = self.departments[best]
        return {"department": d["name"], "email": d.get("email"),
                "phone": d.get("phone"), "url": d.get("url")}

    @staticmethod
    def describe(contact: dict) -> str:
        details = [v for v in (contact.get("email"), contact.get("phone")) if v]
        return contact["department"] + (f" ({' / '.join(details)})" if details else "")
