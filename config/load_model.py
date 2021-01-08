from tensorflow import keras
import sys

model = keras.models.load_model('./models/{}'.format(sys.argv[1]))
model.summary()

## Works
