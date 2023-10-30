# Use an official Python runtime as a parent image
FROM python:3.11.4-slim

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE

# Install pipenv
RUN pip --no-cache-dir install pipenv

# Set the working directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock to the container
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --deploy --system

# Copy the source code, data and models into the container
COPY ./src /app/src
COPY ./data /app/data
COPY ./models /app/models

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "src.prediction_service:app"]
