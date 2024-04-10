FROM shovalmishal/full-pipeline:base
COPY . /workspace
WORKDIR /workspace
RUN pip install -e ./single_image_bg_detector
RUN pip install -e ./resnet_pytorch_small_images

CMD ["/bin/bash", "-c", "chmod +x run_me.sh && ./run_me.sh"]
