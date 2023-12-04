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
# DeepSeek-6.7B-Chat
 
This Space demonstrates model [DeepSeek-Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) by DeepSeek, a code model with 6.7B parameters fine-tuned for chat instructions.
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
    print(message,chat_history,system_prompt,max_new_tokens,temperature,top_p,top_k,repetition_penalty)
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
 
 
def bot(history):
    user_message = history[-1][0]
    new_user_input_ids = tokenizer.encode(
        user_message + tokenizer.eos_token, return_tensors="pt"
    )
 
    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor([]), new_user_input_ids], dim=-1)
 
    # generate a response
    response = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    ).tolist()
 
    # convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(response[0]).split("<|endoftext|>")
    response = [
        (response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)
    ]  # convert to tuples of list
    history[-1] = response[0]
    return history
 
tokens=""
def setvalue1(input):
    tokens = input
 
with gr.Blocks() as demo:
    # chatbot = gr.Chatbot()
    # msg = gr.Textbox()
    # clear = gr.Button("Clear")
    # inp = gr.Slider(
    #             label="Top-p (nucleus sampling)",
    #             minimum=0.05,
    #             maximum=1.0,
    #             step=0.05,
    #             value=0.9,
    #         )
    with gr.Row():
        with gr.Column(scale=1):
            inp_0 = gr.Textbox(label="System prompt", lines=6)
            inp_1 = gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=MAX_MAX_NEW_TOKENS,
                step=1,
                value=DEFAULT_MAX_NEW_TOKENS,
            )
            inp_2 = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=4.0,
                step=0.1,
                value=0,
            )
            inp_3 = gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.9,
            )
            inp_4 = gr.Slider(
                label="Top-k",
                minimum=1,
                maximum=1000,
                step=1,
                value=50,
            )
            inp_5= gr.Slider(
                label="Repetition penalty",
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1,
            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")
 
    msg.submit(generate, [msg, chatbot,inp_0,inp_1,inp_2,inp_3,inp_4,inp_5], [msg, chatbot,inp_0,inp_1,inp_2,inp_3,inp_4,inp_5], queue=False).then(
        bot, chatbot, chatbot
    )
demo.launch()
 
 
 
 
# import os
# from threading import Thread
# from typing import Iterator
 
# import gradio as gr
# import spaces
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
 
# MAX_MAX_NEW_TOKENS = 2048
# DEFAULT_MAX_NEW_TOKENS = 1024
 
# MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
 
# DESCRIPTION = """\
# # DeepSeek-6.7B-Chat
 
# This Space demonstrates model [DeepSeek-Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) by DeepSeek, a code model with 6.7B parameters fine-tuned for chat instructions.
# """
 
# if not torch.cuda.is_available():
#     DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
 
# if torch.cuda.is_available():
#     model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
#     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     tokenizer.use_default_system_prompt = False
 
# @spaces.GPU
# def generate(
#     message: str,
#     chat_history: list,
#     system_prompt: str,
#     max_new_tokens: int = 1024,
#     temperature: float = 0.6,
#     top_p: float = 0.9,
#     top_k: int = 50,
#     repetition_penalty: float = 1,
# ) -> Iterator[str]:
#     conversation = []
#     if system_prompt:
#         conversation.append({"role": "system", "content": system_prompt})
#     for user, assistant in chat_history:
#         conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
#     conversation.append({"role": "user", "content": message})
 
#     input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
#     if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
#         input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
#         gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
#     input_ids = input_ids.to(model.device)
 
#     streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
#     generate_kwargs = dict(
#         input_ids=input_ids,
#         streamer=streamer,
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         num_beams=1,
#         repetition_penalty=repetition_penalty,
#         eos_token_id=32021
#     )
#     t = Thread(target=model.generate, kwargs=generate_kwargs)
#     t.start()
 
#     outputs = []
#     for text in streamer:
#         outputs.append(text)
#         yield "".join(outputs).replace("", "")
 
# def bot(history):
#     user_message = history[-1][0]
#     new_user_input_ids = tokenizer.encode(
#         user_message + tokenizer.eos_token, return_tensors="pt"
#     )
 
#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([torch.LongTensor([]), new_user_input_ids], dim=-1)
 
#     # generate a response
#     response = model.generate(
#         bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
#     ).tolist()
 
#     # convert the tokens to text, and then split the responses into lines
#     response = tokenizer.decode(response[0]).split("\n")  # Provide a non-empty separator for split
#     response = [
#         (response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)
#     ]  # convert to tuples of list
#     history[-1] = response[0]
#     return history
 
# tokens = ""  # Define tokens as a global variable
 
# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column(scale=1):
#             inp_0 = gr.Textbox(label="System prompt", lines=6)
#             inp_1 = gr.Slider(
#                 label="Max new tokens",
#                 minimum=1,
#                 maximum=MAX_MAX_NEW_TOKENS,
#                 step=1,
#                 value=DEFAULT_MAX_NEW_TOKENS,
#             )
#             inp_2 = gr.Slider(
#                 label="Temperature",
#                 minimum=0,
#                 maximum=4.0,
#                 step=0.1,
#                 value=0,
#             )
#             inp_3 = gr.Slider(
#                 label="Top-p (nucleus sampling)",
#                 minimum=0.05,
#                 maximum=1.0,
#                 step=0.05,
#                 value=0.9,
#             )
#             inp_4 = gr.Slider(
#                 label="Top-k",
#                 minimum=1,
#                 maximum=1000,
#                 step=1,
#                 value=50,
#             )
#             inp_5 = gr.Slider(
#                 label="Repetition penalty",
#                 minimum=1.0,
#                 maximum=2.0,
#                 step=0.05,
#                 value=1,
#             )
#     with gr.Column(scale=4):
#         chatbot = gr.Chatbot()
#         msg = gr.Textbox()
#         clear = gr.Button("Clear")
 
#     msg.submit(generate, [msg, chatbot, inp_0, inp_1, inp_2, inp_3, inp_4, inp_5], queue=False).then(
#         bot, chatbot, chatbot
#     )
 
# demo.launch()
