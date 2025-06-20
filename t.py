from __future__ import annotations

from delta_stream import JsonStreamParser
from openai import OpenAI
from pydantic import BaseModel


class Todo(BaseModel):
    task: str
    is_boring: bool | None



aggregated_str = ""

stream_parser = JsonStreamParser(data_model=Todo)

task_str = '{"task":"study","is_boring": true}'
for chunk in task_str:
    result: Todo | None = stream_parser.parse_chunk(chunk)
    
    aggregated_str += chunk
    print(f"'{aggregated_str}' -> {result}")

# class ShortArticle(BaseModel):
#     title: str
#     description: str
#     key_words: list[str]

# # Initialize the stream parser with your Pydantic model
# # Delta stream will try to initialize reasonable defaults for your model, see defaults section
# stream_parser = JsonStreamParser(data_model=ShortArticle)

# client = OpenAI()

# with client.beta.chat.completions.stream(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "Write short articles with a 1-sentence description."},
#         {"role": "user", "content": "Write an article about why it's worth keeping moving forward."},
#     ],
#     response_format=ShortArticle,
# ) as stream:
#     for event in stream:
#         if event.type == "content.delta" and event.parsed is not None:
#             parsed: ShortArticle | None = stream_parser.parse_chunk(event.delta)

#             # If no valuable information was added by the delta
#             # (e.g the LLM is writing a key within the json) 'parsed' will be None
#             if parsed is None:
#                 continue

#             # Valid ShortArticle object, with stream defaults
#             print(parsed)
