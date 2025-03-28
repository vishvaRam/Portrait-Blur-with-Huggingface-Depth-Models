FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /app

COPY Code .

RUN apt-get update

RUN apt-get install -y libgl1 libglib2.0-0

RUN pip install -r requirements.txt

CMD ["python", "main.py"]