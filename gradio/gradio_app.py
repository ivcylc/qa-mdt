# Code by Jade Choghari - Hugging Face - Gradio - 2024
import gradio as gr
import os
import shutil
import spaces
import sys

# Comment this section if you have the dependencies installed
os.system('pip install -r gradio/requirements.txt')
os.system('pip install xformers==0.0.26.post1')
os.system('pip install torchlibrosa==0.0.9 librosa==0.9.2')
os.system('pip install -q pytorch_lightning==2.1.3 torchlibrosa==0.0.9 librosa==0.9.2 ftfy==6.1.1 braceexpand')
os.system('pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121')

## end of section

# only then import the necessary modules from qa_mdt
from qa_mdt.pipeline import MOSDiffusionPipeline


pipe = MOSDiffusionPipeline()

# this runs the pipeline with user input and saves the output as 'awesome.wav'
@spaces.GPU(duration=120)
def generate_waveform(description):
    high_quality_description = "high quality " + description
    pipe(high_quality_description)

    generated_file_path = "./awesome.wav"

    if os.path.exists(generated_file_path):
        return generated_file_path
    else:
        return "Error: Failed to generate the waveform."


intro = """
# 🎶 OpenMusic: AI-Powered Music Diffusion 🎶

Welcome to **OpenMusic**, a next-gen diffusion model designed to generate high-quality audio from text descriptions! 

Simply enter a description of the music you'd like to hear, and our AI will generate it for you.

### Powered by:

- [GitHub](https://github.com/ivcylc/qa-mdt) [@changli](https://github.com/ivcylc) 🎓.
- [Paper](https://arxiv.org/pdf/2405.15863)
- [HuggingFace](https://huggingface.co/jadechoghari/qa_mdt) [@jadechoghari](https://github.com/jadechoghari) 🤗.

Will take 1 to 2 mins to generate 🎶
---

"""

# gradio interface
iface = gr.Interface(
    fn=generate_waveform,
    inputs=gr.Textbox(lines=2, placeholder="Enter a music description here..."),
    outputs=gr.Audio(label="Download the Music 🎼"),
    description=intro,
    examples=[
        ["A modern synthesizer creating futuristic soundscapes."],
        ["Acoustic ballad with heartfelt lyrics and soft piano."]
        ],
    cache_examples=True
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
