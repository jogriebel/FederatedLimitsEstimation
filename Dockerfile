FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV GIT_PYTHON_REFRESH=quiet
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip && \
    python3.11 -m pip install --no-cache-dir --upgrade pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# WORKDIR /app

# ENV GIT_PYTHON_REFRESH=quiet
# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y software-properties-common curl && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get update && apt-get install -y python3.11 python3.11-dev && \
#     curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
#     ln -sf /usr/bin/python3.11 /usr/bin/python && \
#     ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
#     rm -rf /var/lib/apt/lists/* && \
#     python -c "import sys; sys.get_int_max_str_digits()"

# COPY requirements.txt .
# RUN python -m pip install --no-cache-dir -r requirements.txt

# COPY . .

# ENV PYTHONPATH=/app