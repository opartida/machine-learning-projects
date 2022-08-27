FROM tensorflow/serving

ENV MODEL_BASE_PATH /models
ENV MODEL_NAME baggage

COPY model/baggage-predictor /models/baggage-predictor

COPY tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh

RUN chmod +x /usr/bin/tf_serving_entrypoint.sh
ENTRYPOINT []
CMD ["usr/bin/tf_serving_entrypoint.sh"]
