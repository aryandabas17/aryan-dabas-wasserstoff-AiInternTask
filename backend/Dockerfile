FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Add current folder to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

