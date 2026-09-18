FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your model and the server. The sample data under data/ is not copied: only
# local_evaluator.py needs it.
COPY dtos.py utils.py answering.py example.py api.py ./

# Downloads the Whisper model into the image at build time rather than on the
# first request — there is no warm-up grace period once the service starts.
RUN python -c "import example"

EXPOSE 9054

CMD ["python", "api.py"]
