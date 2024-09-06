import os
os.environ['NEED_TO_LOAD_MODEL'] = "True"
import numpy as np
import cv2
import io
from io import BytesIO
import gradio as gr
import random
import time
from PIL import Image
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(current_dir)
from facefusion.core import run
import facefusion.globals

from tqdm import tqdm
def swap_video(source_image, target_video):
    print(type(source_image))
    print(type(target_video))
    target_path = "tmp/target.mp4"
    source_path = "tmp/source.png"
    result_path = "tmp/source.mp4"
    os.system(f"cp {target_video} {target_path}")
    os.system(f"cp {target_video} {target_path}")
    cv2.imwrite(source_path, source_image)

    facefusion.globals.source_path = source_path
    facefusion.globals.target_path = target_path
    facefusion.globals.output_path = result_path
    # 推理
    run()
    return result_path

if __name__ == "__main__":
    global_holder = {}

    with gr.Blocks() as demo:
        gr.Markdown("E4S: Fine-Grained Face Swapping via Regional GAN Inversion")

        with gr.Tab("Video"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    image3_input = gr.Image(label="source image")
                    video_input = gr.Video(label="target video")
                with gr.Column(scale=3):
                    video_output = gr.Video(label="result")
                    video_button = gr.Button("Run")

        video_button.click(
            swap_video,
            inputs=[image3_input, video_input],
            outputs=video_output,
        )

    demo.launch(share=True)

