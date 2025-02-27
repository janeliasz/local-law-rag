FROM python:3.12

ARG UID

WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN useradd -m -u $UID myuser && chown -R myuser /app

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

USER myuser
