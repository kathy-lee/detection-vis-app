from pydantic import BaseModel, ConfigDict
from typing import Optional

class UserCreate(BaseModel):
    email: str
    name: str

class User(BaseModel):
    id: int
    email: str
    name: str

    class ConfigDict:
        from_attributes=True

class Directory(BaseModel):
    id: int
    name: str
    description: str
    parent_id: Optional[int]

    class ConfigDict:
        from_attributes=True


class File(BaseModel):
    id: int
    name: str
    description: str
    size: str
    path: str
    ADC: bool
    RAD: bool
    RA: bool
    AD: bool
    RD: bool
    spectrogram: bool
    radarPC: bool
    lidarPC: bool
    image: bool
    depth_image: bool
    config: str  
    parse: str
    labeled: bool
    directory_id: int
    
    class ConfigDict:
        from_attributes=True


class MLModel(BaseModel):
    id: int
    name: str
    description: str
    # flow_run_id: str
    # flow_name: str
    parent_id: Optional[int]

    class ConfigDict:
        from_attributes=True


class MLModelCreate(BaseModel):
    name: str
    description: str
    # flow_run_id: str
    # flow_name: str
    parent_id: Optional[int] = None
