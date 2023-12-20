# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./main.py /code/main.py
COPY ./preprocess.py /code/preprocess.py
COPY ./train.py /code/train.py
COPY ./amazonproducts.parquet /code/amazonproducts.parquet

RUN python preprocess.py --raw_data_path ./amazonproducts.parquet --dest_path ./
RUN python train.py --data_path ./ --dest_path ./

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
