import numpy
import numpy as np

from keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# 加载训练集
y_train = np.load("data/y.npy")
x_train = np.load("data/x.npy")
x_pre = np.load("data/x_pre.npy")

m_in = Input(shape=11)
d_1 = Dense(11, activation='relu')(m_in)
d_2 = Dense(8, activation='relu')(d_1)

d_3 = Dense(1)(d_2)
model = Model(m_in, d_3)
model.summary()
# model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
# history = model.fit(x_train, y_train, batch_size=32, epochs=10000)
# with open("train.txt","w") as f:
#     for i in history.history["mae"]:
#         f.write(str(i)+"\n")
# pre = list(model.predict(x_train)) + list(model.predict(x_pre))
# pre = numpy.array(pre)
# pre = pre.ravel()
# np.save("pre", pre)
#
