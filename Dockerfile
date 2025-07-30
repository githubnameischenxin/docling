FROM python:3.13-slim-bookworm

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir -r requirements.txt

# On container environments, always set a thread budget to avoid undesired thread congestion.
ENV OMP_NUM_THREADS=4

EXPOSE 8800

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]