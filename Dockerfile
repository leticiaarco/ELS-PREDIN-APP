# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables to optimize Python output and prevent .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://host.docker.internal:11434


# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies (required for compiling packages or image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leveraging Docker layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application using Gunicorn (production WSGI server) or Flask
CMD ["python", "app.py"]