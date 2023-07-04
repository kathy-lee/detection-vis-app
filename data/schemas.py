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

class Directory(BaseModel):
    id: int
    name: str
    description: str
    parent_id: Optional[int]

    class Config:
        orm_mode = True


class File(BaseModel):
    id: int
    name: str
    description: str
    size: str
    path: str
    directory_id: int
    

    class Config:
        orm_mode = True