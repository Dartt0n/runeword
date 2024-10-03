from litestar import Litestar, Router
from litestar.logging.config import StructLoggingConfig
from litestar.plugins.structlog import (
    StructlogConfig,
    StructlogPlugin,
)
from transformers.pipelines import Pipeline

import api
import lib

config = lib.inference.InferenceConfig.from_env()
pipeline = lib.inference.load_pipeline(config.model, config.device, config.dtype)


async def pipeline_di() -> Pipeline:
    return pipeline


app = Litestar(
    route_handlers=[
        Router("/api/v1/health", route_handlers=[api.v1.HealthController]),
        Router("/api/v1/inference", route_handlers=[api.v1.InferenceController]),
    ],
    plugins=[
        StructlogPlugin(
            StructlogConfig(
                structlog_logging_config=StructLoggingConfig(
                    log_exceptions="always",
                )
            )
        )
    ],
    dependencies={"pipeline": pipeline_di},
)
