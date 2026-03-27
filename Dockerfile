# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for caching)
COPY requirements-docker.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the rest of the project files
COPY . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]