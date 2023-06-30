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
