# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev 
    

# Copy poetry.lock and pyproject.toml
COPY poetry.lock pyproject.toml /app/

# Install dependencies
RUN poetry install

# Copy the rest of the application code
COPY . /app

# Run the application
CMD ["poetry", "run", "python", "src/main.py"]