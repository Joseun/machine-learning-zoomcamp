#!/usr/bin/python3
""" MODULE FOR LABELLING MRI IMAGES """
import os
import traceback
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from decouple import config
from PIL import Image
from prefect import flow
from prefect_azure import AzureBlobStorageCredentials
from prefect_azure.blob_storage import blob_storage_download, blob_storage_list

from MRIModel import MRIModel

blob_storage_credentials = AzureBlobStorageCredentials(
    connection_string=config("BLOB"),
)


@flow(name="Get Images", log_prints=True)
async def download_images(blob: str):
    """Download patient images from Azure Blob Storage"""
    image_labels = {
        "ID": [],
        "x_cent": [],
        "y_cent": [],
        "width": [],
        "height": [],
        "plane": [],
    }

    planes = [
        "axial",
        "coronal",
        "sagittal",
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = MRIModel()

    path2weights = "models/weights_smoothl1.pt"
    model.load_state_dict(torch.load(path2weights))

    model = model.to(device)

    try:
        lstdata = await blob_storage_list(
            container=config("CONTAINER_NAME"),
            blob_storage_credentials=blob_storage_credentials,
            name_starts_with=blob,
        )
        print(len(lstdata))
        if len(lstdata) > 0:
            for image in lstdata:
                if image.name.endswith(".jpg") or image.name.endswith(".png"):
                    data = await blob_storage_download(
                        container=config("CONTAINER_NAME"),
                        blob=image.name,
                        blob_storage_credentials=blob_storage_credentials,
                    )
                    print(image.name)
                    path = Path(f"./{image.name}/")
                    path.parents[0].mkdir(parents=True, exist_ok=True)
                    with open(f"./{path}", "wb") as my_blob:
                        my_blob.write(data)
                    print(f"{image.name} downloaded to local from Blob Storage")
                    img = Image.open(image.name)
                    img = TF.to_tensor(img)
                    label_pred = model(img.unsqueeze(0).to(device))[0].cpu()

                    image_labels["ID"].append(image.name.split("/")[-1].split(".")[0])
                    image_labels["x_cent"].append(float(label_pred[0]))
                    image_labels["y_cent"].append(float(label_pred[1]))
                    image_labels["width"].append(float(label_pred[2]))
                    image_labels["height"].append(float(label_pred[3]))
                    image_labels["plane"].append(planes[label_pred[-3:].argmax(0)])
                    os.remove(image.name)
            images_df = pd.DataFrame(image_labels)
            images_df.to_json(f"{blob}/labels.json", orient="records", indent=4)
        else:
            print("No files uploaded")
    except:
        traceback.print_exc()
