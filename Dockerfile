FROM continuumio/miniconda3

# Setting up work directory
WORKDIR /FSDS_

# Copying all files
COPY . .

# Install system-level dependencies
RUN apt-get update && apt-get install -y build-essential

# Copying Conda environment
COPY env.yml .
RUN conda env create -f env.yml

SHELL ["conda","run","-n","fsds-env","/bin/bash","-c"]

RUN pip install -e .

# Setting MLflow to use the local file-based tracking
ENV MLFLOW_TRACKING_URI=file:./mlruns

# Running the scoring script by default
CMD ["conda", "run", "-n", "fsds-env", "env", "PYTHONPATH=src", "python", "main.py", "outputs/"]
