# Use the official Continuum Analytics image for Anaconda
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the conda environment file into the container
COPY conda_environment.yml .

# Create the conda environment
RUN conda env create -f conda_environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "housing", "/bin/bash", "-c"]

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=api.main

# Expose the port that the Flask app runs on
EXPOSE 5003

# Command to run the Flask app
CMD ["conda", "run", "--no-capture-output", "-n", "housing", "flask", "run", "--host=0.0.0.0", "--port=5003"]
