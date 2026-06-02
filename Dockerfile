FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY submission/ submission/

ENV POLICY=submission.my_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder

CMD ["python", "train.py"]
