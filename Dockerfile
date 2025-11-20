FROM python:3.13
RUN mkdir app
WORKDIR /app
COPY *.py .
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
CMD ["python", "main.py"]
