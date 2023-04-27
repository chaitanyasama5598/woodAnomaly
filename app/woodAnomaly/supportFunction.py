import tensorflow as tf
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, model_from_json

### Loading saved model
jFile = open('cnn_autoencoder_24032023'+'.json', 'r')
loaded_model_json = jFile.read()
jFile.close()
autoencoder = model_from_json(loaded_model_json)
autoencoder.load_weights('cnn_autoencoder_24032023'+'.h5')
### Data preparation for modeling
anoImageList = []
imgArray = imgArray.reshape((1, imageSize))
imgArray = imgArray/255
anoImageList.append(imgArray.tolist()[0])
anoImageDf = pd.DataFrame(anoImageList)
xTestCnnCat = np.array(anoImageDf)
xTestCnnCat= np.reshape(xTestCnnCat, (len(xTestCnnCat), 128, 128, 1))
## Making predictions
reconstructions = autoencoder.predict(xTestCnnCat)
test_loss = tf.keras.losses.mae(reconstructions.reshape((1, 128*128)), xTestCnnCat.reshape((1, 128*128)))
testPred = [1 if i>0.19452723240978956 else 0 for i in test_loss]