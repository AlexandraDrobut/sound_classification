def extract_features(file_name):

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None

    return mfccsscaled

import pandas as pd
import librosa
import numpy as np
from keras.models import load_model

data = extract_features('/home/alexandra/Downloads/doberman-pincher_daniel-simion.wav')
#data = extract_features('/home/alexandra/sound-classification/UrbanSound8K/audio/fold10/102857-5-0-20.wav')
print(data.shape)
data = data.reshape(1, data.shape[0], 1)
print(data.shape)
# Load model

model = load_model('/home/alexandra/sound-classification/UrbanSound8K/model_dir/basic_cnn.h5')
model.summary()
weight = model.predict(data)
print(weight)
index = weight.argmax()

class_labels = ["dog_bark", "children_playing", "car_horn", "air_conditioner", "street_music",
				"siren", "engine_idling", "gun_shot", "drilling", "jackhammer"]

class_labels.sort()
print(class_labels[index])