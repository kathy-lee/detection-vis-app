import uuid
#import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI, Depends
from fastapi import UploadFile
#from PIL import Image
from typing import List
from sqlalchemy.orm import Session

from data import crud, models, schemas
from data.database import SessionLocal, engine


# models.Base.metadata.create_all(bind=engine)

# # Create a new session
# db = SessionLocal()

# # Define the user data
# user_data = schemas.UserCreate(email="eleventhhuser@example.com", name="lulu")

# # Call the create_user function to insert the user
# crud.create_user(db=db, user=user_data)

# # Close the session
# db.close()



app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the Detection API"}


@app.post("/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    # image = np.array(Image.open(file.file))
    # model = config.STYLES[style]
    # output, resized = inference.inference(model, image)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    # cv2.imwrite(name, output)
    return {"name": name}


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db)
    return users



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
