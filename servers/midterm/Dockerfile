# docker built -t <name> .
# docker run -it --rm -p 9696:9696 <name>:latest

FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "dv.bin", "options.bin", "./"]

RUN mkdir templates/
COPY "index.html" "templates/"

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
