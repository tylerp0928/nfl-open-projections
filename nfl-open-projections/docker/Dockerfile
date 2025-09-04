FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run extended pipeline for a recent range
CMD ["python","-m","src.pipelines.run_extended","--seasons","2019-2025"]
