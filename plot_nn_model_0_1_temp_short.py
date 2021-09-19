import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from pandas import DataFrame
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from numpy import savetxt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical 
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
import dill as pickle
from sklearn.pipeline import Pipeline
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import learning_curve
from sklearn.ensemble import StackingClassifier
from keras.layers import Input
from keras.models import Model
from keras.losses import binary_crossentropy
from tensorflow.keras import regularizers

# load array
X_train_whole = loadtxt('d:\\table_of_flashes.csv', delimiter=',')

# augment data
choice = X_train_whole[:, -1] == 0.
X_total = numpy.append(X_train_whole, X_train_whole[choice, :], axis=0)
print(X_total.shape)

# balancing
sm = SMOTE(random_state = 2)
X_train_res, Y_train_res = sm.fit_resample(X_total, X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res == 0)))

X_train, X_test, Y_train, Y_test = train_test_split(X_train_res, Y_train_res, random_state=1, test_size=0.3, shuffle = True)
print(X_train.shape)
print(X_test.shape)


#=======================================
 
# Model configuration

input = preprocessing.minmax_scale(X_train[:, 0:152], axis=1)
savetxt('d:\\input.csv', input, delimiter=',')

testinput = preprocessing.minmax_scale(X_test[:, 0:152], axis=1)
savetxt('d:\\testinput.csv', testinput, delimiter=',')
#=====================================

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 4, 38)
input = input.transpose(0, 2, 1)
print (input.shape)
input2d = input[0,:,:]
savetxt('d:\\input_reshaped.csv', input2d, delimiter=',')


testinput = testinput.reshape(len(testinput), 4, 38)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)


# Create the model
model=Sequential()
model.add(Conv1D(filters=35, kernel_size=4, padding='valid', activation='relu', strides=1, input_shape=(38, 4)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=18, kernel_size=2, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))


model.summary()


# Compile the model
sgd = SGD(lr=0.01, momentum=0.7, nesterov=True)       # Higher values get steep loss curves ..
model.compile(loss=sparse_categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

hist = model.fit(input, Y_train, batch_size=32, epochs=400, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None)	


# evaluate the model
Y_hat_classes = model.predict_classes(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()

pyplot.show()

#==================================

model.save("D:\\model.h5")

#==================================

#Removed dropout and reduced momentum and reduced learning rate
