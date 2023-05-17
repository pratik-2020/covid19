import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.optimizers import Adam

class Covid19:
  def _init_(
      self,
      img_pt
  ):
    self.img_pt = img_pt
    self.train_dt = self.gen_dt()
    self.model = self.build_model()
    self.train_model()
  
  def gen_dt(self):
    train = ImageDataGenerator(rescale=1/255)
    train_dt = train.flow_from_directory(
        self.img_pt,
        target_size = (300, 300),
        batch_size=12,
        class_mode="binary"
    )
    return train_dt
  
  def build_model(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation="tanh", input_shape=(300, 300, 3)),
        # tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16, (1,1), activation="tanh"),
        tf.keras.layers.Conv2D(8, (5,5), activation="tanh"),
        tf.keras.layers.Conv2D(16, (1,1), activation="tanh"),
        tf.keras.layers.Conv2D(8, (3,3), activation="tanh"),
        tf.keras.layers.Conv2D(16, (1,1), activation="tanh"),
        tf.keras.layers.MaxPooling2D(3,3),
        tf.keras.layers.Conv2D(16, (1,1), activation="tanh"),
        tf.keras.layers.Conv2D(8, (5,5), activation="tanh"),
        tf.keras.layers.Conv2D(16, (1,1), activation="tanh"),
        tf.keras.layers.Conv2D(8, (3,3), activation="tanh"),
        tf.keras.layers.Conv2D(16, (1,1), activation="tanh"),
        tf.keras.layers.MaxPooling2D(3,3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="tanh"),
        tf.keras.layers.Dense(512, activation="tanh"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.summary()
    return model
  
  def train_model(self):
    self.model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    self.model.fit(
        self.train_dt,
        epochs=15,
        steps_per_epoch=8,
        verbose=1
    )

    return
  
  def predict_img(
      self,
      img_pth
  ):
    img = load_img(img_pth, target_size=(300, 300))
    x = img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])
    return self.model.predict(image, batch_size=10)