# Use an official Python base image (with minimal footprint) for the build stage
FROM prefecthq/prefect:2-latest

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install the required packages
RUN python3 -m pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt --trusted-host pypi.python.org --no-cache-dir

# Copy the rest of the application code into the container at /app
# Copy the rest of the application code into the container at /app
COPY ./main.py ./main.py
COPY ./startapp.sh ./startapp.sh


# Expose the port that the Prefect server will be running on
EXPOSE 4200


# Run the command to start the Prefect agent for automations and scheduling

RUN chmod +x ./startapp.sh

# CMD ["uvicorn", "main:app", "--port", "5000"]

ENTRYPOINT ["bash", "startapp.sh"]
