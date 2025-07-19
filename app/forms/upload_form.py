from fastapi import Request


class UploadForm:
    def __init__(self, request: Request) -> None:
        self.request: Request = request
        self.errors: list = []
        self.model_id: str = ""

    async def load_data(self):
        form = await self.request.form()
        self.model_id = form.get("model_id")

    def is_valid(self):
        if not self.model_id or not isinstance(self.model_id, str):
            self.errors.append("A valid model id is required")
        if not self.errors:
            return True
        return False
