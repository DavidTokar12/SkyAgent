from __future__ import annotations

import base64

from abc import ABC
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from pydantic import BaseModel


class _MessageRole(str, Enum):
    system = "system"
    assistant = "assistant"
    user = "user"
    tool = "tool"


class _BaseMessage(BaseModel, ABC):
    role: _MessageRole

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def _truncate(self, text: str, max_length: int = 124) -> str:
        """Helper method to truncate text with ellipsis if needed"""
        return text if len(text) <= max_length else text[:max_length] + "..."


class SystemMessage(_BaseMessage):
    "Represents a system message in a chat history."
    role: _MessageRole = _MessageRole.system

    def __str__(self) -> str:
        truncated_content = self._truncate(self.content)
        return f"{self.__class__.__name__}(content='{truncated_content}')"

    def __repr__(self):
        return self.__str__()


class AssistantMessage(_BaseMessage):
    "Represents an assistant message in a chat history."
    role: _MessageRole = _MessageRole.assistant
    content: str

    def __str__(self) -> str:
        truncated_content = self._truncate(self.content)
        return f"{self.__class__.__name__}(content='{truncated_content}')"

    def __repr__(self):
        return self.__str__()


# TODO content as string not good. Separate input and output, and probably should be dicts.
class ToolCallOutgoingMessage(_BaseMessage):
    role: _MessageRole = _MessageRole.tool
    content: str
    tool_call_id: str

    def __str__(self) -> str:
        truncated_content = self._truncate(self.content)
        return f"{self.__class__.__name__}(content='{truncated_content}')"

    def __repr__(self):
        return self.__str__()


class ImageAttachment(BaseModel):
    base_64: str
    file_name: str

    @classmethod
    def from_file_path(cls, file_path: Path | str) -> ImageAttachment:
        file_path = Path(file_path)
        with open(file_path, "rb") as file:
            base_64 = base64.b64encode(file.read()).decode("utf-8")
        return cls(base_64=base_64, file_name=file_path.name)

    def __str__(self) -> str:
        return f"Image(file_name='{self.file_name}')"

    def __repr__(self):
        return self.__str__()


class UserTextMessage(_BaseMessage):
    role: _MessageRole = _MessageRole.user
    content: str | None = None

    def __str__(self) -> str:
        truncated_content = self._truncate(self.content)
        return f"{self.__class__.__name__}(content='{truncated_content}')"

    def __repr__(self):
        return self.__str__()


class UserImageMessage(_BaseMessage):
    role: _MessageRole = _MessageRole.user
    base_64: str | None = None

    def __str__(self) -> str:
        return "Image()"

    def __repr__(self):
        return self.__str__()


class FileAttachment(BaseModel):
    file_name: str
    original_file_type: str
    text_content: str
    extracted_images: list[ImageAttachment]

    def __str__(self) -> str:
        return f"File(file_name='{self.file_name}')"

    def __repr__(self):
        return f"File(file_name='{self.file_name}')"


class UserMessage(_BaseMessage):
    role: _MessageRole = _MessageRole.user
    content: str | None = None

    attached_images: list[ImageAttachment] = []
    attached_files: list[FileAttachment] = []

    def __str__(self) -> str:
        parts = []

        if self.content is not None:
            truncated_content = self._truncate(self.content)
            parts.append(f"content='{truncated_content}'")

        if self.attached_images:
            images_str = f"attached_images=[{', '.join(str(img) for img in self.attached_images)}]"
            parts.append(images_str)

        if self.attached_files:
            files_str = f"attached_files=[{', '.join(str(file) for file in self.attached_files)}]"
            parts.append(files_str)

        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __repr__(self):
        return self.__str__()

    def to_text_and_image_messages(self) -> list[UserTextMessage | UserImageMessage]:
        result = []

        result.append(UserTextMessage(content=self.content))

        for image in self.attached_images:
            result.append(UserTextMessage(content=f"Attached image: {image.file_name}"))
            result.append(UserImageMessage(base_64=image.base_64))

        for file in self.attached_files:

            result.append(
                UserTextMessage(
                    content=f"Extracted text content from {file.file_name}: {file.text_content}"
                )
            )

            for index, image in enumerate(file.extracted_images, start=1):
                result.append(
                    UserTextMessage(
                        content=f"Extracted image {index} from {image.file_name}"
                    )
                )
                result.append(UserImageMessage(base_64=image.base_64))

        return result
