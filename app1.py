import os
from threading import Thread
from typing import Iterator

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

DESCRIPTION = """\
# Code Generator
"""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


if torch.cuda.is_available():
    model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False



@spaces.GPU
def generate(
    message: str,
    chat_history: list,
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        eos_token_id=32021
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs).replace("<|EOT|>","")


chat_interface = gr.ChatInterface(
    fn=generate
)

# columns
with gr.Blocks() as demo:
 
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""# Cigniti AI"""),
            gr.Markdown("Cigniti code generation application leverages GenAI/LLMs to assist the programers in writing the optmized code.")
            gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=MAX_MAX_NEW_TOKENS,
                step=1,
                value=DEFAULT_MAX_NEW_TOKENS,
            ),
            gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.9,
            ),
            gr.Slider(
                label="Top-k",
                minimum=1,
                maximum=1000,
                step=1,
                value=50,
            ),
            gr.Slider(
                label="Repetition penalty",
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1,
            ),     
        with gr.Column(scale=4):
            gr.Markdown(DESCRIPTION)
            chat_interface.render()



# with gr.Blocks(css="style.css") as demo:
#     gr.Markdown(DESCRIPTION)
#     chat_interface.render()

if __name__ == "__main__":
    demo.queue().launch(share=True)
