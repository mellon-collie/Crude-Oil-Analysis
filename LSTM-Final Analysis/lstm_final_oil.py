from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np


def p_change_conv(s):
    #s = s.split('"')
    #s = s[1]
    s = s.split("..")
    if(len(s)<2):
        #print(s)
        return float(s[0])
    k = [i for i in s[0]]
    m = [i for i in s[1]]
    f = ''.join(k[1:])+"."+''.join(m[:-1])
    return float(f)

def conv_to_float(n):
    if type(n)==int or type(n)==float:
        return float(n)
    l = n.split('.')
    if(len(l)==1):
        return float(l[0])
    m = l[1]+"."+l[-2]
    return float(m)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
'''
df = read_csv('final_oil.csv', header=0, index_col=0)
a = list(df["Dollar_eq"])
a = list(map(conv_to_float,a))
a.reverse()
b = list(df["Volume"])
b.reverse()
c = list(df["Percent_Change"])
c = list(map(p_change_conv,c))
c.reverse()
d = list(df["Price"])
d.reverse()

df1 = DataFrame({"Dollar_eq":a,"Volume":b,"Percent_Change":c,"Price":d})
#print(df1.head())
'''

df1 = read_csv('final_oil.csv', header=0, index_col=0)
df1 = df1[["Price","US_Price","Dollar_eq","Open","High","Low","Volume","Percent_Change"]]
print(df1.head())
values = df1.values

values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
#print(reframed.head())
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
#print(values)
split = int(len(list(df1["Dollar_eq"]))*0.8)
train = values[:split,:]
test = values[split:,:]
train_X,train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
#print(test_y)

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)



pyplot.plot(range(1,len(yhat)+1),inv_y,label='actual')
pyplot.plot(range(1,len(yhat)+1),inv_yhat,label='predicted')
pyplot.legend()
pyplot.show()












