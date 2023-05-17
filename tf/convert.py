import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from util import IterDataset as dataset

classes = 19

# model architecture
model = Sequential()
model.add(SimpleRNN(50))
model.add(Dense(classes, activation='softmax',
                activity_regularizer=tf.keras.regularizers.l1(0.00001)))

opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
_STD = 10
data = dataset(b"../data/pre_processed2.csv", _STD, 50)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
checkpoint.restore(tf.train.latest_checkpoint('.tmp/model'))

model.fit(
    data,
    epochs=1,
    verbose=1,
    steps_per_epoch=5
)
print("Saving model")
model.save('model')
print("Model saved")

def _data():
    for _ in range(1000):
        yield [data.__next__()[0]]

print("Converting to tflite")
converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = _data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("running converter")
tflite_quant_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
