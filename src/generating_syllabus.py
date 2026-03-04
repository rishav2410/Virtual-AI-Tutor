from gemini_client import generate_text


def generate_syllabus(topic: str, task: str) -> str:
    """Generate a concise, structured syllabus for the given topic using Gemini."""
    prompt = (
        "You are an expert course designer. "
        f"Create a practical syllabus for: {topic}. "
        f"Context task: {task}.\\n\\n"
        "Return a clean markdown syllabus with:\\n"
        "1) Course overview\\n"
        "2) Learning outcomes\\n"
        "3) Module-by-module plan (ordered)\\n"
        "4) Suggested exercises and mini-projects\\n"
        "5) 2-week and 4-week study tracks\\n"
        "Keep it specific and beginner-friendly."
    )

    try:
        return generate_text(prompt)
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
