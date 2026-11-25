"""Request and response models for establishing a voice connection."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VoiceSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID


class VoiceSessionResponse(BaseModel):
    url: str
    room_name: str
    token: str = Field(repr=False)
