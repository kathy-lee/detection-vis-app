from sqlalchemy.orm import Session
from . import models, schemas

def get_users(db: Session):
    return db.query(models.User).all()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(name=user.name, email=user.email)  # Convert UserCreate to User
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_datasets(db: Session):
    # return db.query(models.Directory).filter(models.Directory.parent_id==None)
    return db.query(models.Directory).all()

def get_datasetfiles(db: Session, dataset_id: int):
    return db.query(models.File).filter(models.File.directory_id == dataset_id).all()

def get_datafile(db: Session, file_id: int):
    return db.query(models.File).filter(models.File.id == file_id).first()

def get_datafile_by_name(db: Session, file_name: str):
    return db.query(models.File).filter(models.File.name == file_name).first()

def add_model(db: Session, mlmodel: schemas.MLModelCreate):
    #model = models.MLmodel(name=mlmodel.name, description=mlmodel.description, flow_run_id=mlmodel.flow_run_id, flow_name=mlmodel.flow_name, parent_id=mlmodel.parent_id)  
    model = models.MLmodel(name=mlmodel.name, description=mlmodel.description, parent_id=mlmodel.parent_id)  
    db.add(model)
    db.commit()
    db.refresh(model)
    return model

def get_models(db: Session):
    return db.query(models.MLmodel).all()

def get_model(db: Session, model_id: int):
    return db.query(models.MLmodel).filter(models.MLmodel.id == model_id).first()