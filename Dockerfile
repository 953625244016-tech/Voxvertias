FROM python:3.10-slim

# Install system audio dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run training to ensure model exists
RUN python train.py

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Create a start script to run both
RUN echo "#!/bin/bash\n uvicorn main:app --host 0.0.0.0 --port 8000 & \n streamlit run frontend/dashboard.py --server.port 8501 --server.address 0.0.0.0" > start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]