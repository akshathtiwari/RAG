# Use a lightweight Python base
FROM python:3.10-slim

# 1) Install Linux tools & Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2) Create app directory & an offload directory
RUN mkdir -p /app/offload
WORKDIR /app

# 3) Copy your current directory (.) into /app, excluding files/folders from .dockerignore
COPY . /app

# 4) Install Python dependencies
#    Make sure your requirements.txt is in your project folder
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5) Expose the default Streamlit port
EXPOSE 8501

# 6) Default command to run Streamlit
CMD ["streamlit", "run", "tessaract.py", "--server.address=0.0.0.0", "--server.port=8501"]
