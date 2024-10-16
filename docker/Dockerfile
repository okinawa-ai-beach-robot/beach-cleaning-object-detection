FROM python:3.9

# Environment for training and evaluating yolov5 object detection models for the beachbot package.

WORKDIR /app

RUN apt-get update && apt-get install -y git

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
RUN git clone https://github.com/okinawa-ai-beach-robot/beach-cleaning-object-detection.git
WORKDIR $HOME/app/beach-cleaning-object-detection
RUN git fetch origin object_detection_metrics && git checkout object_detection_metrics
RUN pip install .

WORKDIR $HOME/app
