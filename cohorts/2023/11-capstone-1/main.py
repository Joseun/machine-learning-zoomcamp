#!/usr/bin/python3
""" ENDPOINT TO PREDICT IMAGE LABELS"""
import asyncio
from datetime import datetime
from pathlib import Path

import uvicorn
from decouple import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prefect_azure import AzureBlobStorageCredentials
from prefect_azure.blob_storage import blob_storage_download
from pydantic import BaseModel, Field

from labeller import download_images

blob_storage_credentials = AzureBlobStorageCredentials(
    connection_string=config("BLOB"),
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/home")
async def home():
    return {"message": "welcome"}


class Study(BaseModel):
    id: int
    name: str = Field(None, title="name of patient", min_length=5)
    date: datetime = Field(None, title="time of study")


@app.post("/process_images/")
async def student_data(payload: Study):
    try:
        blob = f"{payload.name}/{payload.date.strftime('%Y%m%dH%HM%M')}"
        asyncio.create_task(download_images(blob), name="Download Images")
        response = {"message": "Success", "data": f"{blob} being processed"}
        return JSONResponse(status_code=200, content=response)
    except HTTPException as error:
        error.detail.update({"status_code": error.status_code})
        return JSONResponse(status_code=200, content=error.detail)
    except Exception as error:
        error_body = {"message": "Server currently busy"}
        print(error)
        raise HTTPException(status_code=500, detail=error_body) from error


if __name__ == "__main__":
    data = blob_storage_download(
        container=config("CONTAINER_NAME"),
        blob="models/weights_smoothl1.pt",
        blob_storage_credentials=blob_storage_credentials,
    )
    path = Path("models/weights_smoothl1.pt")
    path.parents[0].mkdir(parents=True, exist_ok=True)
    with open(f"./{path}", "wb") as my_blob:
        my_blob.write(data)
    print(f"model weights downloaded to local from Blob Storage")

    print("Starting fast API server")
    uvicorn.run("main:app", reload=True, port=5000, workers=2)
