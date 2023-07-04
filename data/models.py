from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship


Base = declarative_base()

# Define a User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

class Directory(Base):
    __tablename__ = "directories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    description = Column(String, index=True)
    parent_id = Column(Integer, ForeignKey("directories.id"))  # Parent directory


    files = relationship("File", back_populates="directory")
    subdirectories = relationship("Directory")  # Subdirectories

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, index=True)
    size = Column(String, index=True)
    path = Column(String, index=True)
    directory_id = Column(Integer, ForeignKey("directories.id"))  # The directory the file is in

    directory = relationship("Directory", back_populates="files")