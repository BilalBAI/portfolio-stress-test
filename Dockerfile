# Use the official Python 3.8 image from Docker Hub as the base image
FROM python:3.8.6

WORKDIR /app

# Application Pip Requirements
COPY requirements-docker.txt ./
RUN pip-sync requirements-docker.txt  && \
    python3.8 -m pip cache purge || true

# Application
COPY . ./
RUN python3.8 -m pip install -e .

ENTRYPOINT ["/bin/bash", "-l"]
