ARG BASE_IMAGE=python:3.10-slim-buster
FROM $BASE_IMAGE
COPY echo_service.py ./
RUN pip install --upgrade pip && \
    pip install orchestration-framework flask
CMD ["python3", "echo_service.py"]
