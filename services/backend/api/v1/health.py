from litestar import Controller, MediaType, get


class HealthController(Controller):
    @get(media_type=MediaType.TEXT, include_in_schema=False)
    async def health(self) -> str:
        return "OK"
