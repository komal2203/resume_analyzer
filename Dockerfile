# Use python 3.11 base image:
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install dependencies:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]