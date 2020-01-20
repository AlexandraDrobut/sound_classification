import pandas as pd
import numpy as np
import os
import librosa

#Return mfccs of every sound file
def extract_features(file_name):
	try:
		audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
		mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
		mfccsscaled = np.mean(mfccs.T,axis=0)

	except Exception as e:
		print("Error encountered while parsing file: ", file)
		return None
	return mfccsscaled

# Iterate through each sound file and extract the features
def parse_dataset():
	# Set the path to the full UrbanSound dataset
	fulldatasetpath = '/home/alexandra/sound-classification/UrbanSound8K/audio'
	metadata = pd.read_csv('/home/alexandra/sound-classification/UrbanSound8K/metadata/UrbanSound8K.csv')
	features = []
	for index, row in metadata.iterrows():

		file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
		class_label = row["class"]
		data = extract_features(file_name)
		features.append([data, class_label])
	return features


# Convert into a Panda dataframe
featuresdf = pd.DataFrame(parse_dataset(), columns=['feature','class_label'])
print('Finished feature extraction from ', len(featuresdf), ' files')

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
feature_vectors = np.array(featuresdf.feature.tolist())
class_label = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
label_enc = LabelEncoder()
class_label_matrix = to_categorical(label_enc.fit_transform(class_label))

# split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature_vectors, class_label_matrix, test_size=0.2, random_state = 42)

#create CNN architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

num_labels = class_label_matrix.shape[1]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Construct model
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(x_train.shape[1], 1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 115
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='model_dir/basic_cnn.h5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)
# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

#Let's see how other classifier evaluate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib import pyplot

models = {
"knn": KNeighborsClassifier(n_neighbors=1),
"naive_bayes": GaussianNB(),
"decision_tree": DecisionTreeClassifier(),
"random_forest": RandomForestClassifier(n_estimators=100),
"mlp": MLPClassifier(),
"svm": SVC(kernel="linear"),
}

results = []
names = []
scoring = "accuracy"

for name in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(models[name], feature_vectors, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
