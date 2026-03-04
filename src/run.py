import gradio as gr

from gemini_client import get_gemini_api_key
from generating_syllabus import generate_syllabus
from teaching_agent import teaching_agent

# Validate key early so startup errors are explicit.
get_gemini_api_key()

with gr.Blocks() as demo:
    gr.Markdown("# Your AI Instructor")

    with gr.Tab("Input Your Information"):

        def perform_task(input_text: str) -> str:
            try:
                task = "Generate a course syllabus to teach the topic: " + input_text
                syllabus = generate_syllabus(input_text, task)
                teaching_agent.seed_agent(syllabus, task)
                return syllabus
            except Exception as exc:
                return f"Error: {exc}"

        text_input = gr.Textbox(label="State the name of topic you want to learn:")
        text_output = gr.Textbox(label="Your syllabus will be shown here:")
        text_button = gr.Button("Build the Bot!!!")
        text_button.click(perform_task, text_input, text_output)

    with gr.Tab("AI Instructor"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="What do you concern about?")
        clear = gr.Button("Clear")

        def user(user_message, history):
            if not user_message or not user_message.strip():
                return "", history or []
            teaching_agent.human_step(user_message)
            history = history or []
            history.append({"role": "user", "content": user_message})
            return "", history

        def bot(history):
            if not history:
                return []
            bot_message = teaching_agent.instructor_step()
            history.append({"role": "assistant", "content": bot_message})
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: [], None, chatbot)

demo.queue().launch(debug=True, share=True)
