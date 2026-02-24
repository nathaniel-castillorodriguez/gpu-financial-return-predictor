FROM python:3.10-slim

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
RUN pip install flask numpy --quiet

COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
