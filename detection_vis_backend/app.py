import uuid
import os
import logging
import uvicorn
import paramiko
import subprocess
import json
import torch
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi import Response, status
#from PIL import Image
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from metaflow import Flow
from metaflow.exception import MetaflowException
from pathlib import Path

# import sys
# sys.path.insert(0, '/home/kangle/projects/detection-vis-app')

from data import crud, schemas
from data.database import SessionLocal
from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.networks.network import NetworkFactory
from detection_vis_backend.train.utils import DisplayHMI
from detection_vis_backend.train.train import train




# models.Base.metadata.create_all(bind=engine)

# # Create a new session
# db = SessionLocal()

# # Define the user data
# user_data = schemas.UserCreate(email="eleventhhuser@example.com", name="lulu")

# # Call the create_user function to insert the user
# crud.create_user(db=db, user=user_data)

# # Close the session
# db.close()

logging.basicConfig(level=logging.INFO)


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
    

@app.post("/train")
async def train_model(datafiles_chosen: list[Any], features_chosen: list[Any], mlmodel_configs: dict, train_configs: dict, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    # 
    try: 
        train(datafiles_chosen, features_chosen, mlmodel_configs, train_configs)
        with open('exp_info.txt', 'r') as f:
            exp_name = f.read()

        model_meta = schemas.MLModelCreate(name=exp_name,description="info",parent=None)
        crud.add_model(db=db, mlmodel=model_meta)
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during training: {str(e)}")

    return {"model_name": exp_name}


@app.get("/models", response_model=List[schemas.MLModel])
def read_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        models = crud.get_models(db)
    except Exception as e:
        logging.error(f"An error occurred during reading the models: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while getting the models.")

    if not models:
        raise HTTPException(status_code=404, detail="No models found.")
    return models


@app.get("/model/{id}")
def read_model(id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        model = crud.get_model(db, id)
        if not model:
            raise ValueError("Model not found")
        flow_name = model.flow_name
        run_id = model.flow_run_id
        run = Flow(flow_name)[run_id]
        parameters = run.data
        
        if not parameters:
            raise ValueError("Parameters are empty")

    except MetaflowException as e:
        print(f"Metaflow exception occurred: {str(e)}")
    except ValueError as e:
        print(f"ValueError occurred: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
            
    return {"datafiles": parameters.datafiles, "features": parameters.features, 
            "model_config": parameters.model_config, "train_config": parameters.train_config}



# @app.get("/save/{model_id}")
# async def save_model(model_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     try:
        
#     except Exception as e:
#         logging.error(f"An error occurred during training the model: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An error occurred during training the model: {str(e)}")
#     return Response(status_code=status.HTTP_204_NO_CONTENT)
    



# @app.get("/evaulate/{model_id}")
# async def evaulate_model(model_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     try:
#         model = crud.get_model(db, model_id)
#     except Exception as e:
#         logging.error(f"An error occurred during evaluating the model: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An error occurred druing evaluating the model: {str(e)}")

#     return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/predict/{model_id}")
async def predict(model_id: int, checkpoint_id: int, sample_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        # Get para infos
        model = crud.get_model(db, model_id)
        if not model:
            raise ValueError("Model not found")
        flow_name = model.flow_name
        run_id = model.flow_run_id
        run = Flow(flow_name)[run_id]
        parameters = run.data

        # Get input data (For now could only handle one datafile case)
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(parameters.datafiles[0]["parse"], parameters.datafiles[0]["id"])
        dataset_inst.parse(parameters.datafiles[0]["path"], parameters.datafiles[0]["name"], parameters.datafiles[0]["config"])
        input_data = dataset_inst[sample_id]

        # Initialize the model
        model_config = parameters.model_config
        network_factory = NetworkFactory()
        model_type = model_config['type']
        model_config.pop('type', None)
        net = network_factory.get_instance(model_type, model_config)

        # Load the model checkpoint
        model_rootdir = os.getenv('MODEL_ROOTDIR')
        model_path = os.path.join(model_rootdir, model.name)
        checkpoint = [file for file in os.listdir(model_path) if f"epoch{checkpoint_id:02}" in file][0]
        dict = torch.load(os.path.join(model_path, checkpoint))
        net.load_state_dict(dict['net_state_dict'])  

        # Prediction
        net.eval()
        # input_data: [radar_FFT, segmap,out_label,box_labels,image]
        input = torch.tensor(input_data[0]).permute(2,0,1).unsqueeze(0)
        with torch.set_grad_enabled(False):
            output = net(input)
        
        # Display and evaluate the output
        obj_labels = input_data[3]
        # Plan to replace DisplayHMI with a new method: 
        # GetBoxes(output, obj_labels)
        # output: 2 lists of boxes-- pred_boxes=[(u1,v1,u2,v2),...]; label_boxes is the same
        print(f"network output shape: {output['Detection'].shape}, {output['Segmentation'].shape}")
        print(f"input image and rd shape: {input_data[4].shape}, {input_data[0].shape}")
        hmi = DisplayHMI(input_data[4], input_data[0], output, obj_labels)
        # pred_obj = output['Detection']
        # obj_pred = np.asarray(decode(pred_obj,0.05))  
        # TP,FP,FN = GetDetMetrics(obj_pred,obj_labels,threshold=0.2,range_min=5,range_max=100)
    except Exception as e:
        logging.error(f"An error occurred during model prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred druing model prediction: {str(e)}")

    return {"prediction": hmi.tolist()} #, "eval": [TP,FP,FN]


@app.get("/predict_newdata/{model_id}")
async def predict_newdata(model_id: int, input_file: UploadFile, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        # Get input data
        contents = await input_file.read()  # read the file
        if input_file.filename.endswith('.npy'):
            input_data = np.load(io.BytesIO(contents))
        elif input_file.filename.endswith(('.jpg', '.png')):
            image = Image.open(io.BytesIO(contents))
            input_data = np.array(image)
        # elif input_file.filename.endswith('.pcd'):  # assuming a point cloud file with '.pcd' extension
        #     # process point cloud data
        #     data = read_pcl(contents)
        else:
            raise ValueError("Unsupported file format")
        

    except Exception as e:
        logging.error(f"An error occurred during model prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred druing model prediction: {str(e)}")

    return None


@app.post("/retrain/{model_id}")
async def retrain_model(model_id: int, datafiles_chosen: list[Any], features_chosen: list[Any], mlmodel_configs: dict, train_configs: dict, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):  
    try: 
        model = crud.get_model(db, model_id)
        train(datafiles_chosen, features_chosen, mlmodel_configs, train_configs, model.name)
        with open('exp_info.txt', 'r') as f:
            exp_name = f.read()

        model_meta = schemas.MLModelCreate(name=exp_name,description="info",parent=model_id)
        crud.add_model(db=db, mlmodel=model_meta)
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during training: {str(e)}")

    return {"model_name": exp_name}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
