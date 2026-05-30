FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN rm -f /etc/pip/constraint.txt \
    && unset PIP_CONSTRAINT \
    && pip install --no-cache-dir --upgrade 'protobuf>=4.25.3,<5' \
    && pip install --no-cache-dir \
    mediapipe \
    opencv-python-headless \
    gradio \
    tritonclient[http,grpc] \
    pyyaml \
    pytest pytest-cov

WORKDIR /app
COPY . /app

CMD ["python", "-m", "src.ui.app"]
