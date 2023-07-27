from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
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
    # raw data type
    ADC = Column(Boolean, index=True)
    RAD = Column(Boolean, index=True)
    RA = Column(Boolean, index=True)
    AD = Column(Boolean, index=True)
    RD = Column(Boolean, index=True)
    spectrogram = Column(Boolean, index=True)
    radarPC = Column(Boolean, index=True)
    lidarPC = Column(Boolean, index=True)
    image = Column(Boolean, index=True)
    depth_image = Column(Boolean, index=True)
    # config file path
    config = Column(String, index=True)
    # parse class
    parse = Column(String, index=True)
    # whether it's labeled
    labeled = Column(Boolean, index=True)
    
    directory_id = Column(Integer, ForeignKey("directories.id"))  # The directory the file is in

    directory = relationship("Directory", back_populates="files")


class MLmodel(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    description = Column(String, index=True)
    flow_run_id = Column(Integer, index=True)
    flow_name = Column(String, index=True)
    # data_input
    # feature_input
    # model_config
    # train_config