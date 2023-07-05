import uuid
#import cv2
import os
import uvicorn
import paramiko
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi import Response, status
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

@app.get("/datasets", response_model=List[schemas.Directory])
def read_datasets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    datasets = crud.get_datasets(db)
    return datasets

@app.get("/dataset/{id}", response_model=List[schemas.File])
def read_datasetfiles(id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    datasetfiles = crud.get_datasetfiles(db, dataset_id = id)
    if datasetfiles is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return datasetfiles

@app.get("/download")
async def download_file(file_path: str, file_name: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically add the server's SSH key (not recommended for production)
    private_key = paramiko.RSAKey.from_private_key_file('id_rsa')
    ssh.connect('mifcom-desktop', username='kangle', pkey=private_key)
    sftp = ssh.open_sftp()
    #remote_file_path = os.path.join("/home/kangle", file_path, file_name)
    remote_file_path = os.path.join("/home/kangle", file_path, "test.txt") # for testing
    local_file_path = file_name
    sftp.get(remote_file_path, local_file_path)
    sftp.close()
    ssh.close()
    return Response(status_code=status.HTTP_204_NO_CONTENT)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
