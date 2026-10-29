from pydantic import BaseModel, Field


class CheckboxBox(BaseModel):
    bbox: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in pixel coordinates of the uploaded image",
    )
    is_checked: bool


class DetectResponse(BaseModel):
    boxes: list[CheckboxBox]
