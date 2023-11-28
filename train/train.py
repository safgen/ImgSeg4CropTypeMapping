import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow_addons as tfa
import tensorflow as tf
from datetime import datetime
import os
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from train.confusion_matrix import plot_confusion_matrix, plot_to_image
from train.UNETModel import unet_model
from deeplabModel import DeepLabV3Plus
from AttnResUnet import Attention_ResUNet

'''
  DATA SET LOADING FUNCTIONS
'''
def parse_tfrecord(example_proto):
  return tf.io.parse_single_example(example_proto, FEATURES_DICT)

def to_tuple(inputs):
  inputsList = [inputs.get(key) for key in FEATURES]
  stacked = tf.stack(inputsList, axis=0)
  stacked = tf.transpose(stacked, [1, 2, 0])
  return stacked[:,:,:len(BANDS)], tf.squeeze(tf.one_hot(indices=int(stacked[:,:,-1]), depth= NCLASS))

def get_dataset(pattern):
	glob = tf.io.gfile.glob(pattern)
	dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
	dataset = dataset.map(parse_tfrecord)
	dataset = dataset.map(to_tuple)
	return dataset

def get_training_dataset():
	glob =  FOLDER + '/' + TRAINING_BASE + '*'
	dataset = get_dataset(glob)
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
	return dataset

def get_eval_dataset():
  glob =  FOLDER + '/' + EVAL_BASE + '*'
  dataset = get_dataset(glob)
  dataset = dataset.batch(BATCH_SIZE).repeat()
  return dataset


# # Specify Config Variables for Data and Model Paths
SOURCE = 's2-data' ## update this for new data source
JOB_FOLDER = 'AttResUnet_FTLoss_g120-s2'  ## update this for new models
JOB_DIR = JOB_FOLDER + '/trainer'
MODEL_DIR = JOB_DIR + '/model'
LOGS_DIR = JOB_DIR + '/logs'
HIST_DIR = JOB_DIR + '/history'
checkpoint_path = JOB_DIR + "/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


FOLDER = SOURCE + '/20190830' ## update this for data period
TRAINING_BASE = 'training_patches'
EVAL_BASE = 'eval_patches'
# 
# # Specify inputs (bands) to the model and the response variable.
if SOURCE == 's2-data':
  opticalBands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
else:
  opticalBands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
thermalBands = []
BANDS = opticalBands  + thermalBands
RESPONSE = 'cropland'
FEATURES = BANDS + [RESPONSE]
NCLASS  =  3
class_names = ['Others', 'Corn', 'Soybean']

# # Specify the size and shape of patches expected by the model.
KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

# # Sizes of the training and evaluation datasets.
TRAIN_SIZE = 20000
EVAL_SIZE = 10000

# Specify model training parameters.
BATCH_SIZE = 32
EPOCHS = 50
BUFFER_SIZE = 6000
OPTIMIZER = 'adam'
# LOSS = 'categorical_crossentropy'

# Create Confusion matrix log directory
logdir_cm = LOGS_DIR + "/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer_cm = tf.summary.create_file_writer(logdir_cm + '/cm')

# LOAD Training and Validation datasets
training = get_training_dataset()
evaluation = get_eval_dataset() 
# CREATE EVALUATION SAMPLES FOR CONFUSION MATRIX
test_images, test_labels = iter(evaluation.take(1)).next()
test_labels = np.argmax(test_labels, axis=3)
  
# SET UP CONTEXT FOR TRAINING WITH THE PREFERED DEVICE STRATEGY
strat = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
with strat.scope():
  from train.metrics import dice_coef, jacard_coef
  # function to log confusion matrices like loss and metrics logs  
  def log_confusion_matrix(epoch, logs):
      
      # Use the model to predict the values from the test_images.
      test_pred_raw = m.predict(test_images)
      
      test_pred = np.argmax(test_pred_raw, axis=3)
      
      # Calculate the confusion matrix using sklearn.metrics
      cm = sklearn.metrics.confusion_matrix(test_labels.reshape(-1,1), test_pred.reshape(-1,1))
      
      figure = plot_confusion_matrix(cm, class_names=class_names)
      cm_image = plot_to_image(figure)
      
      # Log the confusion matrix as an image summary.
      with file_writer_cm.as_default():
          tf.summary.image("Confusion Matrix", cm_image, step=epoch)

  # LIST of Evaluation Metrics, Only uncomment the required ones
  METRICS = [ 
            tf.keras.metrics.CategoricalAccuracy(name='CategoricalAccuracy'), 
            tf.keras.metrics.MeanIoU(NCLASS),
            # tf.keras.metrics.Accuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            # tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR'),
            tfa.metrics.F1Score(NCLASS, average='micro'),
            dice_coef, 
            jacard_coef
            ]

  # Tversky Loss defined as Custom Loss Function (alpha=beta=0.5 => Dice Loss)
  def tversky_loss(y_true, y_pred, alpha = 0.5, beta  = 0.5):
            
      ones = tf.ones(tf.shape(y_true))
      p0 = y_pred      # proba that voxels are class i
      p1 = ones-y_pred # proba that voxels are not class i
      g0 = y_true
      g1 = ones-y_true
      
      num = tf.math.reduce_sum(p0*g0, (0,1,2))
      den = num + alpha*tf.math.reduce_sum(p0*g1,(0,1,2)) + beta*tf.math.reduce_sum(p1*g0,(0,1,2))
      
      T = tf.math.reduce_sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
      
      Ncl = tf.cast(tf.shape(y_true)[-1], 'float32')
      return Ncl-T
  
  # Focal Tversky Loss (to customize focus on easier or harder examples) defined as Custom Loss Function (alpha=beta=0.5 => Dice Loss)
  def focal_tversky_loss(y_true, y_pred, gamma=1.5):
    tv = tversky_loss(y_true, y_pred)
    return tf.math.pow(tv, gamma)

  # m = unet_model(BANDS, NCLASS)
  # m = DeepLabV3Plus(KERNEL_SIZE, KERNEL_SIZE, len(BANDS), NCLASS)
  m = Attention_ResUNet((KERNEL_SIZE, KERNEL_SIZE, len(BANDS)),  NUM_CLASSES=NCLASS)
  print(m.summary())
  m.compile(
    optimizer=optimizers.get(OPTIMIZER), 
    loss=focal_tversky_loss,
    # loss = losses.get(LOSS),
    metrics=[metric for metric in METRICS])

  # LOAD Training and Validation datasets
  training = get_training_dataset()
  evaluation = get_eval_dataset()
  
  # Checkpoint callback for every 5 epochs
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=2, 
    save_weights_only=True,
    save_freq=10*(TRAIN_SIZE//BATCH_SIZE))
  
  # Per-epoch callback for confusion matriux.  
  tensorboard_cm_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir_cm, histogram_freq = 1)
  cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
  
  #check if the saved checkpoints exist
  latest = tf.train.latest_checkpoint(checkpoint_dir)

  # Load weights if there was a checkpoint
  if latest:
    m.load_weights(latest)

  m.save_weights(checkpoint_path.format(epoch=0))
  # START TRAINING
  history = m.fit(
      x=training,
      epochs=EPOCHS, 
      steps_per_epoch=int(TRAIN_SIZE / BATCH_SIZE), 
      validation_data=evaluation,
      validation_steps=int(EVAL_SIZE / BATCH_SIZE),
      validation_freq=1,
      callbacks=[tf.keras.callbacks.TensorBoard(LOGS_DIR), cp_callback,  cm_callback],
    )
  m.save(MODEL_DIR, save_format='tf')
  # save history into a CSV file
  hist_df = pd.DataFrame(history.history) 
  os.mkdir(HIST_DIR)
  hist_csv_file = HIST_DIR +'/' + str(EPOCHS) + '-epochs.csv'
  with open(hist_csv_file, "w+") as f:
    hist_df.to_csv(f)
  
  

