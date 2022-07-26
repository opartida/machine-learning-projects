import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  
  # Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
  _VOCAB_SIZE = 1000
  # Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
  _OOV_SIZE = 10
  # Number of buckets used by tf.transform for encoding each feature.
  _FEATURE_BUCKET_COUNT = 10

  _FEATURE_KEYS = ['DEPARTURE', 
                   'ADULTS', 
                   'CHILDREN', 
                   'INFANTS', 
                   'ARRIVAL', 
                   'TRIP_TYPE', 
                   'TRAIN', 
                   'GDS', 
                   'HAUL_TYPE',
                   'NO_GDS',
                   'WEBSITE',
                   'PRODUCT',
                   'SMS',
                   'DISTANCE']

  _VOCAB_FEATURE_KEYS = ['DEPARTURE', 
                         'ARRIVAL', 
                         'EXTRA_BAGGAGE', 
                         'TRIP_TYPE', 
                         'TRAIN', 
                         'HAUL_TYPE',
                         'WEBSITE',
                         'PRODUCT',
                         'SMS']

  _CATEGORICAL_FEATURE_KEYS = ['ADULTS', 'CHILDREN', 'INFANTS', 'GDS', 'NO_GDS']

  _DENSE_FLOAT_FEATURE_KEYS = ['DISTANCE']

  _BUCKET_FEATURE_KEYS = []

  _LABEL_KEY = 'EXTRA_BAGGAGE'
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
              inputs[key], 
              _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[key] = inputs[key]  

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
