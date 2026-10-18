FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir scanpy leidenalg igraph pyarrow

COPY cluster_imyoo.py /opt/imyoo/cluster_imyoo.py
COPY cluster_study.py /opt/imyoo/cluster_study.py
WORKDIR /opt/imyoo

ENTRYPOINT ["python", "/opt/imyoo/cluster_imyoo.py"]
