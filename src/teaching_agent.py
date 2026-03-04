import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class TeachingGPT:
    """Simple instructor agent that teaches from a generated syllabus."""
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.syllabus = ""
        self.conversation_topic = ""
        self.conversation_history: list[str] = []
    def seed_agent(self, syllabus: str, task: str) -> None:
        self.syllabus = syllabus
        self.conversation_topic = task
        self.conversation_history = []
    def human_step(self, human_input: str) -> None:
        self.conversation_history.append(f"User: {human_input}")
    def instructor_step(self) -> str:
        if not self.syllabus:
            return "Please generate a syllabus first from the first tab."
        history = "\n".join(self.conversation_history[-12:])
        prompt = (
            "You are a helpful AI instructor. Teach step-by-step using the syllabus.\n\n"
            f"Topic/task: {self.conversation_topic}\n\n"
            f"Syllabus:\n{self.syllabus}\n\n"
            "Conversation so far:\n"
            f"{history if history else 'No prior messages.'}\n\n"
            "Respond with one focused teaching step, then ask one short check-for-understanding question."
        )
        response = self.llm.invoke(prompt)
        ai_message = response.content if hasattr(response, "content") else str(response)
        self.conversation_history.append(f"Instructor: {ai_message}")
        return ai_message
def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or export it in your shell."
        )
_require_openai_key()
teaching_agent = TeachingGPT(llm=ChatOpenAI(temperature=0.7))
