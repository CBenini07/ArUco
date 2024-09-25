# Base da imagem (ajuste a versão do Python se necessário)
FROM python:3.9-slim-buster  

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "detector_video.py"]