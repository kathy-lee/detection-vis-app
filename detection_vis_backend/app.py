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
from detection_vis_backend.dataset import DatasetFactory, RaDICaL, RADIal


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


@app.get("/parse/{file_id}")
async def parse_data(file_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        datafile = crud.get_datafile(db, file_id)
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(datafile.parse, file_id)
        dataset_inst.parse(datafile.path, datafile.name, datafile.config)
    except Exception as e:
        logging.error(f"An error occurred during parsing the raw data file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while parsing the raw data file: {str(e)}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/sync/{file_id}")
async def get_sync(file_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    try:
        datafile = crud.get_datafile(db, file_id)
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(datafile.parse, file_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to retrieve sync mode.")

    return dataset_inst.frame_sync


@app.get("/feature/{file_id}/{feature_name}/size")
async def get_feature_size(file_id: int, feature_name: str,  skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    try:
        datafile = crud.get_datafile(db, file_id)
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(datafile.parse, file_id)

        if feature_name in ("RD", "RA", "spectrogram", "radarPC"):
            count = dataset_inst.radarframe_count
        elif feature_name == "lidarPC":
            count = dataset_inst.lidarframe_count
        elif feature_name == "image":
            count = dataset_inst.image_count
        else:
            count = dataset_inst.depthimage_count
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to retrieve feature size.")

    return count
 

@app.get("/feature/{file_id}/{feature_name}/{id}")
async def get_feature(file_id: int, feature_name: str, id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    try:
        datafile = crud.get_datafile(db, file_id)
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(datafile.parse, file_id)
        function_dict = {
            'RAD': dataset_inst.get_RAD,
            'RD': dataset_inst.get_RD,
            'RA': dataset_inst.get_RA,
            'spectrogram': dataset_inst.get_spectrogram,
            'radarPC': dataset_inst.get_radarpointcloud,
            'lidarPC': dataset_inst.get_lidarpointcloud,
            'image': dataset_inst.get_image,
            'depth_image': dataset_inst.get_depthimage,
        }
        feature = function_dict[feature_name](id)

        if feature_name == "RAD":
            serialized_feature = [[[(x.real, x.imag) for x in y] for y in z] for z in feature.tolist()]
        else:
            serialized_feature = feature.tolist()
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Feature ID {id} is out of range.")

    return {"serialized_feature": serialized_feature}
    

@app.get("/train/{model_id}")
async def train_model(model_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    # try:
    #     model = crud.get_model(db, model_id)
    #     model_factory = DatasetFactory()
    #     model_inst = model_factory.get_instance(model.type, model_id)
        

    # Kick off a run of training flow with specified parameters
    flow = Flow('TrainModelFlow')
    try:
        run = flow.run(parameters={'datafiles': datafiles_chosen,
                                   'features': features_chosen,
                                   'model_config': model_configs,
                                   'train_config': train_configs})
        run.wait_for_completion()
        accuracy = run['end'].task.data.accuracy
    except MetaflowException as e:
        print(f"Metaflow exception occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    except Exception as e:
        logging.error(f"An error occurred during training the model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while training the model: {str(e)}")

    return Response(status_code=status.HTTP_204_NO_CONTENT)



@app.get("/save/{model_id}")
async def save_model(model_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        model = crud.add_model(db, model_id)
    except Exception as e:
        logging.error(f"An error occurred during training the model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during training the model: {str(e)}")

    return Response(status_code=status.HTTP_204_NO_CONTENT)



@app.get("/evaulate/{model_id}")
async def evaulate_model(model_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        model = crud.get_model(db, model_id)
    except Exception as e:
        logging.error(f"An error occurred during evaluating the model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred druing evaluating the model: {str(e)}")

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/predict/{model_id}")
async def predict(model_id: int, input_data: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        model = crud.get_model(db, model_id)
        prediction = model.predict(input_data)
    except Exception as e:
        logging.error(f"An error occurred during evaluating the model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred druing evaluating the model: {str(e)}")

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
