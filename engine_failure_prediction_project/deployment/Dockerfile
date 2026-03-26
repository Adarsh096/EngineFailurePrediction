# Use a minimal base image with Python 3.10 installed
FROM python:3.10-slim

#Install neccesary libraries:
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

#Below ones don't work in huggingface streamlit spaces:
#ENV HUGGINGFACE_USER_NAME=${HUGGINGFACE_USER_NAME}
#ENV HUGGINGFACE_MODEL_NAME=${HUGGINGFACE_MODEL_NAME}

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Expose container port for Streamlit:
EXPOSE 7860

# Define the command to run the Streamlit app on port "7860" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--gatherUsageStats=false"]
