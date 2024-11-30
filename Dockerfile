FROM python:3.11-slim    
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY . .

RUN pip install -r requirements.txt
EXPOSE 5050
CMD ["python", "app.py"]

