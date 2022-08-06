
from typing import List
from absl import logging
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.tfxio import dataset_options

logging.set_verbosity(logging.ERROR)  # Set default logging level.

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
_VOCAB_SIZE = 1000
# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
_OOV_SIZE = 10
# Number of buckets used by tf.transform for encoding each feature.
_FEATURE_BUCKET_COUNT = 10

_FEATURE_KEYS = ['WEBSITE','GDS','DEPARTURE','ARRIVAL',
    'ADULTS','CHILDREN','INFANTS','TRAIN','HAUL_TYPE',
    'DISTANCE','DEVICE','TRIP_TYPE','PRODUCT','SMS','NO_GDS'
]
_LABEL_KEY = 'EXTRA_BAGGAGE'

_VOCAB_FEATURE_KEYS = ['DEPARTURE','ARRIVAL', 'WEBSITE','GDS', 
                       'TRAIN', 'TRIP_TYPE', 'DEVICE', 'SMS', 
                       'NO_GDS', 'HAUL_TYPE', 'PRODUCT', 'EXTRA_BAGGAGE']

_CATEGORICAL_FEATURE_KEYS = ['ADULTS','CHILDREN','INFANTS']

_DENSE_FLOAT_FEATURE_KEYS = ['DISTANCE']

_BUCKET_FEATURE_KEYS = []

def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      tf_transform_output.transformed_metadata.schema).repeat()


def _build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(2):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(1, activation='sigmoid')(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(1e-2),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.Accuracy()])

  model.summary(print_fn=logging.info)
  return model

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
    outputs[key] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[key] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[key] = tft.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[key] = _fill_in_missing(inputs[key])
  
  

  return outputs

  
def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """  
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  
  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=_TRAIN_BATCH_SIZE)  

  model = _build_keras_model()
  
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_steps=fn_args.eval_steps)

  # The result of the training should be saved in `fn_args.serving_model_dir`
  # directory.
  model.save(fn_args.serving_model_dir, save_format='tf')
