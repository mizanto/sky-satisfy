# Use an official Python runtime as a parent image
FROM python:3.11.4-slim

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE

# Set the working directory in the container
WORKDIR /app

# Install pipenv
RUN pip --no-cache-dir install pipenv

# Copy Pipfile and Pipfile.lock to the container
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --deploy --system

# Copy the source code, data and models into the container
COPY ./backend /app/backend
COPY ./frontend /app/frontend
COPY ./data /app/data
COPY ./models /app/models

# Expose the port the app runs on
EXPOSE 8000

# Define gunicorn as the entry point, with recommended number of workers
ENTRYPOINT ["gunicorn", "--workers=3", "--bind", "0.0.0.0:8000", "backend.prediction_service:app"]

# HEALTHCHECK to ensure '/health' endpoint responds OK, or container is marked unhealthy.
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl --fail http://localhost:8000/health || exit 1
