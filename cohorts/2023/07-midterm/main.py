#!/usr/bin/python3
""" ENDPOINT tTO GENERATE RECOMMENDATIONS"""
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting fast API server")


ratings = pd.read_parquet(r"./amazonproducts.parquet")
recommendations = pd.read_parquet(r"./recommendations.parquet")


@app.get("/home")
async def home():
    return {"message": "welcome"}


@app.post("/recommend/{UserID}")
async def recommed_products(UserID: str):
    """Recommend products"""

    try:
        if UserID not in recommendations.userId.unique():
            # print("Please select")
            # print(recommendations.userId)
            prod_rating = ratings.groupby("ProductId").filter(
                lambda x: x["Score"].count() >= 50
            )
            data = list(
                prod_rating.groupby("ProductId")["Score"]
                .mean()
                .sort_values(ascending=False)
                .head()
                .to_dict()
                .keys()
            )
            response = {"message": "Success", "data": data}
            return JSONResponse(status_code=200, content=response)
        else:
            data = recommendations[recommendations.userId == UserID][
                "productId"
            ].to_list()
            response = {"message": "Success", "data": data}
            # print(response)
            return JSONResponse(status_code=200, content=response)
    except HTTPException as error:
        error.detail.update({"status_code": error.status_code})
        return JSONResponse(status_code=200, content=error.detail)
    except Exception as error:
        error_body = {"message": "Server currently busy"}
        print(error)
        raise HTTPException(status_code=500, detail=error_body) from error


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=5000, workers=2)
