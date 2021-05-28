FROM tensorflow/tensorflow:2.5.0-gpu

ARG USER_ID=1001
ARG GROUP_ID=1001

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


COPY . /workspace/

RUN python -m pip install --upgrade pip
RUN pip install -r /workspace/repo/requirements.txt


WORKDIR /workspace
RUN chown user -R /workspace

ENV TFDS_DATA_DIR=/workspace/tensorflow_datasets

USER user

CMD "python" "app.py"