import os
import warnings

import msgspec
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines import Pipeline

warnings.filterwarnings("ignore")


class InferenceConfig(msgspec.Struct):
    model: str
    device: str
    dtype: torch.dtype

    @classmethod
    def from_env(cls):
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        model = os.environ["MODEL"]

        return cls(model=model, device=device, dtype=dtype)


def load_pipeline(model_name: str, device: str, dtype: torch.dtype) -> Pipeline:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=".cache/model",
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=".cache/processor",
    )

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )


class InferenceRequest(msgspec.Struct):
    audio: np.ndarray
    sampling_rate: int


def perform_inference_task(pipeline: Pipeline, msg: InferenceRequest) -> str:
    result = pipeline({"raw": msg.audio, "sampling_rate": msg.sampling_rate})
    return result["text"]  # type: ignore
