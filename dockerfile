FROM python:3.11-slim

WORKDIR /app

COPY ./app /app
COPY ./templates /app/templates
COPY ./static /app/static
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=flask_api.py
ENV FLASK_ENV=production

CMD ["python", "flask_api.py"]
