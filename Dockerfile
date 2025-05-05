FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY ./GR-MG /app/GR-MG

RUN sh /app/GR-MG/goal_gen/install.sh