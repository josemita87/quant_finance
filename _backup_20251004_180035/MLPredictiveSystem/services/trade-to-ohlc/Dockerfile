FROM python:3.12-slim

#Add src to the PYTHONPATH
ENV PYTHONPATH=/app

ENV PYTHONBUFFERED=1

# Install Poetry
RUN pip install poetry==1.8.3

# Set the working directory
WORKDIR /app

# Copy only the poetry files first to leverage caching
COPY pyproject.toml poetry.lock /app/

# Install dependencies
RUN poetry install

# Copy the rest of the application code
COPY . /app

# Run the application
CMD ["poetry", "run", "python", "src/main.py"]