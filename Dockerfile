FROM python:3.11-slim

# create non-root user
RUN useradd -m appuser
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

USER appuser
CMD ["uvicorn", "quant_pipeline.app:app", "--host", "0.0.0.0", "--port", "8000"]
