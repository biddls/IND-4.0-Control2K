import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbMetricsLogger
import os
from util import IterDataset as dataset
from tqdm import tqdm
os.environ["WANDB_SILENT"] = "true"
import wandb
classes = 19

# Start a run, tracking hyperparameters
runID = wandb.init(project="David AI IOT-tf", resume=True)
print(f"{runID.id=}")

# model architecture
model = Sequential()
model.add(SimpleRNN(50))
model.add(Dense(classes, activation='softmax',
                activity_regularizer=tf.keras.regularizers.l1(0.00001)))


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.acc = [0, 0]
        self.__std = _STD
        self.threshold = 0.95
        wandb.define_metric("custom_step")
        wandb.define_metric(
            "epoch/std", step_metric="custom_step")
        log_dict = {
            "epoch/std": self.__std,
            "custom_step": 0,
        }
        wandb.log(log_dict)
        self.prog_bar = tqdm(range(epochs), unit="Epoch", dynamic_ncols=True, initial=wandb.run.step)

    def on_epoch_end(self, _epoch, logs=None):
        self.prog_bar.update(1)
        self.prog_bar.set_description(
            f"Acc {100 * logs['accuracy']:.2f}% - std: {self.__std:.2f}")
        self.acc.append(logs['accuracy'])
        if self.acc[-1] > self.threshold and self.acc[-2] > self.threshold:
            self.__std *= 0.99
            data.send(self.__std)
            log_dict = {
                "epoch/std": self.__std,
                "custom_step": int(_epoch),
            }
            wandb.log(log_dict)
        manager.save()
        model.save('model')


BATCH_SIZE = 100
EPOCHS = 20_000
opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
_STD = 1.638
data = dataset(b"../data/pre_processed2.csv", _STD, 50)
model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
manager = tf.train.CheckpointManager(
        checkpoint,
        directory="./tmp/model",
        max_to_keep=1,
    )
status = checkpoint.restore(manager.latest_checkpoint)

model.fit(
        data,
        epochs=EPOCHS,
        verbose=0,
        initial_epoch=wandb.run.step,
        steps_per_epoch=BATCH_SIZE,
        callbacks=[
                CustomCallback(EPOCHS),
                WandbMetricsLogger(log_freq="epoch"),
            ],
        max_queue_size=40,
)


