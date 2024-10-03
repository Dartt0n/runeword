import os

import gradio as gr
import numpy as np
import requests


class App(gr.Interface):
    def __init__(self):
        super().__init__(
            fn=App.greet,
            inputs=[
                gr.Audio(label="Audio"),
                gr.Number(label="Sample Rate", value=16000),
            ],
            outputs=[
                gr.Textbox(label="Transcription"),
            ],
            title="RuneWord",
            css="footer{display:none !important}",
            allow_flagging="never",
        )

    @staticmethod
    def greet(audio: tuple[int, np.ndarray], sample_rate: int) -> str:
        audio_data = audio[1]
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        r = requests.post(
            f"http://{os.environ['BACKEND_URL']}/api/v1/inference",
            json={"audio": audio_data.tolist(), "sample_rate": sample_rate},
        )
        r.raise_for_status()

        return r.json()["text"]
