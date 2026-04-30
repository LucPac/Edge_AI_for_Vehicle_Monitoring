FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tencent.com/pypi/simple/

COPY . .

RUN chmod +x run_all.sh

CMD ["./run_all.sh"]
