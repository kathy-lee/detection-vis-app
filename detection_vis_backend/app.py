import uuid
import os
import logging
import uvicorn
import paramiko
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi import Response, status
#from PIL import Image
from typing import List
from sqlalchemy.orm import Session

# import sys
# sys.path.insert(0, '/home/kangle/projects/detection-vis-app')

from data import crud, models, schemas
from data.database import SessionLocal, engine
from detection_vis_backend.processing import DatasetFactory, RaDICaL, RADIal


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
    try:
        users = crud.get_users(db)
    except Exception as e:
        logging.error(f"An error occurred during getting the users: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while getting the users.")

    if not users:
        raise HTTPException(status_code=404, detail="No users found.")
    return users


@app.get("/datasets", response_model=List[schemas.Directory])
def read_datasets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        datasets = crud.get_datasets(db)
    except Exception as e:
        logging.error(f"An error occurred during reading the datasets: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while getting the datasets.")

    if not datasets:
        raise HTTPException(status_code=404, detail="No datasets found.")
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
    try:
        # private_key = paramiko.RSAKey.from_private_key_file("id_rsa")
        private_key = paramiko.RSAKey.from_private_key_file("/home/kangle/.ssh/id_rsa_mifcom")
    except paramiko.PasswordRequiredException:
        print("Private key is encrypted, password is required")
    except paramiko.SSHException:
        print("Private key file is not a valid RSA private key file")
    except FileNotFoundError:
        print("Private key file does not exist")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    try:
        ssh.connect('mifcom-desktop', username='kangle', pkey=private_key)
    except paramiko.AuthenticationException:
        return Response(content="SSH connection failed", status_code=status.HTTP_401_UNAUTHORIZED)
    
    sftp = ssh.open_sftp()
    remote_file_path = os.path.join("/home/kangle/dataset", file_path, file_name)
    # remote_file_path = os.path.join("/home/kangle", file_path, "test.txt") # for testing
    local_file_path = file_name
    # sftp.get(remote_file_path, local_file_path)
    # sftp.close()
    # ssh.close()
    try:
        sftp.get(remote_file_path, local_file_path)
    except IOError:
        return Response(content="File not found", status_code=status.HTTP_404_NOT_FOUND)
    finally:
        sftp.close()
        ssh.close()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/parse")
async def parse_data(parser: str, file_path: str, file_name: str, config: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(parser)
        dataset_inst.parse(file_path, file_name, config)
    except Exception as e:
        logging.error(f"An error occurred during parsing the raw data file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while parsing the raw data file: {str(e)}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/feature/{feature_name}/size")
async def get_feature_size(parser: str, feature_name: str,  skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    try:
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(parser)

        function_dict = {
            'RAD': dataset_inst.get_RAD,
            'RD': dataset_inst.get_RD,
            'RA': dataset_inst.get_RA,
            'AD': dataset_inst.get_AD,
            'spectrogram': dataset_inst.get_spectrogram,
            'radarPC': dataset_inst.get_radarpointcloud,
            'lidarPC': dataset_inst.get_lidarpointcloud,
            'image': dataset_inst.get_image,
            'depth_image': dataset_inst.depth_image,
        }
        feature = function_dict[feature_name]()
        count = len(feature)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Featue {feature_name} doesn't exist.")

    return count
    

@app.get("/feature/{feature_name}/{id}")
async def get_feature(parser: str, feature_name: str, id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    try:
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(parser)
        function_dict = {
            'RAD': dataset_inst.get_RAD,
            'RD': dataset_inst.get_RD,
            'RA': dataset_inst.get_RA,
            'AD': dataset_inst.get_AD,
            'spectrogram': dataset_inst.get_spectrogram,
            'radarPC': dataset_inst.get_radarpointcloud,
            'lidarPC': dataset_inst.get_lidarpointcloud,
            'image': dataset_inst.get_image,
            'depth_image': dataset_inst.depth_image,
        }
        features = function_dict[feature_name]()
        feature = features[id]

        if feature_name == "RAD":
            serialized_feature = [[[(x.real, x.imag) for x in y] for y in z] for z in feature.tolist()]
        else:
            serialized_feature = feature.tolist()
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Feature ID {id} is out of range.")

    return {"serialized_feature": serialized_feature}
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
