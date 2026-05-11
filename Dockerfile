FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN pip install --no-cache-dir \
    mediapipe \
    opencv-python-headless \
    gradio \
    tritonclient[http,grpc] \
    pyyaml \
    pytest pytest-cov

WORKDIR /app
COPY . /app

CMD ["python", "-m", "src.ui.app"]
