from pydantic import BaseModel


class TextInput(BaseModel):
    text: str


class SearchResult(BaseModel):
    content: str
    url: str
    distance: float


class UrlDocumentInput(BaseModel):
    content: str
    url: str