import gradio as gr
from llm.airllm_loader import generate_response
from tts.tts_engine import speak

def chat(user_input):
    response = generate_response(user_input)
    audio_path = speak(response)
    return response, audio_path

gr.Interface(
    fn=chat,
    inputs="text",
    outputs=["text", "audio"],
    title="Julius Voice LLM"
).launch()
