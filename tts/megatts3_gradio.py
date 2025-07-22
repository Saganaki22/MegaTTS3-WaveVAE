# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import torch
import os
from functools import partial
import gradio as gr
import traceback
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav


def model_worker(input_queue, output_queue, device_id):
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = input_queue.get()
        inp_audio_path, inp_text, infer_timestep, p_w, t_w = task
        try:
            convert_to_wav(inp_audio_path)
            wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
            cut_wav(wav_path, max_len=28)
            with open(wav_path, 'rb') as file:
                file_content = file.read()
            resource_context = infer_pipe.preprocess(file_content, latent_file=None)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            output_queue.put(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(task, str(e))
            output_queue.put(None)


def main(inp_audio, inp_text, infer_timestep, p_w, t_w, processes, input_queue, output_queue):
    print("Push task to the inp queue |", inp_audio, inp_text, infer_timestep, p_w, t_w)
    input_queue.put((inp_audio, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        # Save the result to a temporary file with .wav extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(res)
            tmp_file.flush()
            return tmp_file.name
    else:
        print("")
        return None


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp_manager = mp.Manager()

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices != '':
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    else:
        devices = None
    
    num_workers = 1
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    processes = []

    print("Start open workers")
    for i in range(num_workers):
        p = mp.Process(target=model_worker, args=(input_queue, output_queue, i % len(devices) if devices is not None else None))
        p.start()
        processes.append(p)

    # Define examples with sample voices and texts
    examples = [
        ["samples/sample-1.mp3", "Hello world, this is a test of speech synthesis using artificial intelligence.", 32, 1.4, 3.0],
        ["samples/sample-2.mp3", "The quick brown fox jumps over the lazy dog in the meadow.", 32, 1.4, 3.0],
        ["samples/sample-3.mp3", "Welcome to MegaTTS3, an advanced text-to-speech system with voice cloning capabilities.", 32, 1.4, 3.0]
    ]

    with gr.Blocks(title="MegaTTS3-WaveVAE") as demo:
        gr.HTML("<h1 style='text-align: center;'>MegaTTS3-WaveVAE</h1>")
        gr.HTML("<p style='text-align: center;'>Upload a speech clip as a reference for voice cloning, input the target text, and receive the synthesized speech in the reference voice.</p>")
        
        with gr.Row():
            with gr.Column():
                inp_audio = gr.Audio(type="filepath", label="Upload Reference Audio 10-25 seconds")
                inp_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter the text you want to convert to speech...")
                infer_timestep = gr.Number(label="Inference Timesteps", value=32, minimum=1, maximum=100)
                p_w = gr.Number(label="Intelligibility Weight (p_w)", value=1.4, minimum=0.5, maximum=5.0, step=0.1)
                t_w = gr.Number(label="Similarity Weight (t_w)", value=3.0, minimum=0.0, maximum=10.0, step=0.1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                output_audio = gr.Audio(label="Synthesized Audio", format="wav")
        
        gr.Examples(
            examples=examples,
            inputs=[inp_audio, inp_text, infer_timestep, p_w, t_w]
        )
        
        # GitHub link at the very bottom
        gr.HTML("<div style='text-align: center; margin-top: 30px; padding: 20px;'><a href='https://github.com/Saganaki22/MegaTTS3-WaveVAE' target='_blank'>Github</a></div>")
        
        submit_btn.click(
            fn=partial(main, processes=processes, input_queue=input_queue, output_queue=output_queue),
            inputs=[inp_audio, inp_text, infer_timestep, p_w, t_w],
            outputs=[output_audio]
        )
        
        clear_btn.click(
            fn=lambda: [None, "", 32, 1.4, 3.0, None],
            outputs=[inp_audio, inp_text, infer_timestep, p_w, t_w, output_audio]
        )

    demo.launch(server_name='0.0.0.0', server_port=7929, debug=True)
    for p in processes:
        p.join()