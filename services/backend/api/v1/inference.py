from typing import Annotated

import msgspec
import numpy as np
from litestar import Controller, post
from litestar.params import Body
from transformers.pipelines import Pipeline

import lib


class InferenceRequest(msgspec.Struct):
    audio: list[float]
    sample_rate: int


class InferenceResponse(msgspec.Struct):
    text: str


class InferenceController(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @post()
    async def inference(
        self,
        data: Annotated[InferenceRequest, Body],
        pipeline: Pipeline,
    ) -> InferenceResponse:
        text = lib.inference.perform_inference_task(
            pipeline,
            lib.inference.InferenceRequest(
                audio=np.array(data.audio),
                sampling_rate=data.sample_rate,
            ),
        )
        return InferenceResponse(text=text)
