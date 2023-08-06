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

from data import crud, models, schemas
from data.database import SessionLocal, engine
from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.networks.network import NetworkFactory
from detection_vis_backend.train.utils import DisplayHMI, GetDetMetrics, decode



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
    try:
        # Convert the parameter values to strings
        datafiles_str = json.dumps(datafiles_chosen)
        logging.error(f"datafiles_chosen:{datafiles_chosen}")
        features_str = json.dumps(features_chosen)
        model_config_str = json.dumps(mlmodel_configs)
        train_config_str = json.dumps(train_configs)

        train_file = Path("detection_vis_backend/train/train.py")
        if train_file.is_file():  
            with open('train_log.txt', 'w') as f:  
                subprocess.run(["python", "detection_vis_backend/train/train.py", "run", 
                                "--datafiles", datafiles_str,
                                "--features", features_str,
                                "--model_config", model_config_str,
                                "--train_config", train_config_str])
            
            # For now Metaflow doesn't have an API for launching flows programmatically, this is the temporary way to get run id and model checkpoint path
            with open('modelflow_info.txt', 'r') as f:
                lines = f.readlines()
            for line in lines:
                if "RUN_ID:" in line:
                    run_id = line.replace("RUN_ID: ", "").strip()
                elif "EXP_NAME:" in line:
                    exp_name = line.replace("EXP_NAME: ", "").strip()

            # #########################For debugging
            exp_name = "FFTRadNet___Aug-03-2023___19:57:20" 
            model_data = schemas.MLModelCreate(name=exp_name,description="info",flow_run_id=run_id, flow_name="TrainModelFlow")
            crud.add_model(db=db, mlmodel=model_data)
        else:
            raise FileNotFoundError(f"{train_file} does not exist")
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found error: {str(fnf_error)}")
        raise HTTPException(status_code=500, detail=f"File not found: {str(fnf_error)}")
    except MetaflowException as e:
        print(f"Metaflow exception occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

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
async def predict(model_id: int, sample_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        # Get para infos
        model_dict = crud.get_model(db, model_id)
        if not model_dict:
            raise ValueError("Model not found")
        flow_name = model_dict["flow_name"]
        run_id = model_dict["flow_id"]
        run = Flow(flow_name)[run_id]
        parameters = run.data

        # Get input data
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(parameters.datafiles[0]["parse"], parameters.datafiles[0]["id"])
        input_data = dataset_inst[sample_id]

        # Initialize the model
        model_config = parameters.model_config
        network_factory = NetworkFactory()
        model_type = model_config['type']
        model_config.pop('type', None)
        model = network_factory.get_instance(model_type, model_config)

        # Load the model checkpoint
        model_rootdir = os.getenv('MODEL_ROOTDIR')
        model_path = os.path.join(model_rootdir, model_dict["name"])
        dict = torch.load(model_path)
        model.load_state_dict(dict['net_state_dict'])  

        # Prediction
        model.eval()
        # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        input = torch.tensor(input_data[0]).permute(2,0,1).to('cuda').float().unsqueeze(0)
        with torch.set_grad_enabled(False):
            output = model(input)

        # Display and evaluate the output
        obj_labels = input_data[3]
        # Plan to replace DisplayHMI with a new method: 
        # GetBoxes(output, obj_labels)
        # output: 2 lists of boxes-- pred_boxes=[(u1,v1,u2,v2),...]; label_boxes is the same
        hmi = DisplayHMI(input_data[4], input_data[0], output)
        pred_obj = output['Detection'].detach().cpu().numpy().copy()
        obj_pred = np.asarray(decode(pred_obj,0.05))  
        TP,FP,FN = GetDetMetrics(obj_pred,obj_labels,threshold=0.2,range_min=5,range_max=100)
    except Exception as e:
        logging.error(f"An error occurred during model prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred druing model prediction: {str(e)}")

    return {"prediction": hmi.tolist(), "eval": [TP,FP,FN]}


@app.get("/predict_newdata/{model_id}")
async def predict(model_id: int, input_file: UploadFile, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        # Get para infos
        model_dict = crud.get_model(db, model_id)
        if not model_dict:
            raise ValueError("Model not found")
        flow_name = model_dict["flow_name"]
        run_id = model_dict["flow_id"]
        run = Flow(flow_name)[run_id]
        parameters = run.data

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
        
        # Initialize the model
        model_config = parameters.model_config
        network_factory = NetworkFactory()
        model_type = model_config['type']
        model_config.pop('type', None)
        model = network_factory.get_instance(model_type, model_config)

        # Load the model checkpoint
        model_rootdir = os.getenv('MODEL_ROOTDIR')
        model_path = os.path.join(model_rootdir, model_dict["name"])
        dict = torch.load(model_path)
        model.load_state_dict(dict['net_state_dict'])  

        # Prediction
        model.eval()
        # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        input = torch.tensor(input_data).permute(2,0,1).to('cuda').float().unsqueeze(0)
        with torch.set_grad_enabled(False):
            output = model(input)

        # Display the output
        # Need to handle multiple files 
        hmi = DisplayHMI(None, input_data, output)
        # pred_obj = output['Detection'].detach().cpu().numpy().copy()
        # obj_pred = np.asarray(decode(pred_obj,0.05))  
        # TP,FP,FN = GetDetMetrics(obj_pred,obj_labels,threshold=0.2,range_min=5,range_max=100)

    except Exception as e:
        logging.error(f"An error occurred during model prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred druing model prediction: {str(e)}")

    return {"prediction": hmi.tolist()}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
