from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    email: str
    name: str

class User(BaseModel):
    id: int
    email: str
    name: str

    class Config:
        orm_mode = True

