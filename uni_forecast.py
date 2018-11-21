import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('final_oil.csv', header=0, index_col=0)

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

df1 = pd.DataFrame({"Dollar_eq":a,"Volume":b,"Percent_Change":c})

def scale(l,a,b):
    l_min = min(l)
    l_max = max(l)
    k = []
    for i in l:
        k.append((b-a)*(i-l_min)/(l_max-l_min) +a)
        
    return k


def ar_model(feature):
    data = df1[feature]
    values = data.values
    values = values.astype('float32')
    split = int(len(list(df1["Dollar_eq"]))*0.8)
    #scaled = scale(values,0,1)
    train = values[:split]
    test = values[split:]
    model = AR(train)
    model_fit = model.fit()
        # make prediction
    yhat = model_fit.predict(len(train),len(train)+len(test )-1)
    x = []
    for i in range(len(test)):
        x.append((yhat[i],test[i]))
    #print(x[0:10])
    rmse = sqrt(mean_squared_error(test,yhat))
    print("\n\n AR RMSE: %.5f" % rmse)
    print()
    

def ma_model(feature):
    data = df1[feature]
    values = data.values
    values = values.astype('float32')
    split = int(len(list(df1["Dollar_eq"]))*0.8)
    #scaled = scale(values,0,1)
    train = values[:split]
    test = values[split:]
    model = ARMA(train, order=(0, 1))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(train),len(train)+len(test )-1)
    x = []
    for i in range(len(test)):
        x.append((yhat[i],test[i]))
    #print(x[0:10])
    rmse = sqrt(mean_squared_error(test,yhat))
    print("\n\n MA RMSE: %.5f" % rmse)
    print()
    
    
#ARMA (p,q) = (2,1)
def arma_model(feature):
    data = df1[feature]
    values = data.values
    values = values.astype('float32')
    split = int(len(list(df1["Dollar_eq"]))*0.8)
    #scaled = scale(values,0,1)
    train = values[:split]
    test = values[split:]
    #(p,q) = (2,1)
    model = ARMA(train, order=(2, 1))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(train),len(train)+len(test )-1)
    x = []
    for i in range(len(test)):
        x.append((yhat[i],test[i]))
    #print(x[0:10])
    rmse = sqrt(mean_squared_error(test,yhat))
    print("\n\n ARMA RMSE: %.5f" % rmse)
    print()

#ARIMA (p,d,q) = (1,1,1)
def arima_model(feature):
    data = df1[feature]
    values = data.values
    values = values.astype('float32')
    split = int(len(list(df1["Dollar_eq"]))*0.8)
    #scaled = scale(values,0,1)
    train = values[:split]
    test = values[split:]
    #(p,d,q) = (1,1,1)
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(train),len(train)+len(test )-1,typ="levels")
    x = []
    for i in range(len(test)):
        x.append((yhat[i],test[i]))
    #print(x[0:10])
    rmse = sqrt(mean_squared_error(test,yhat))
    print("\n\n ARIMA RMSE: %.5f" % rmse)
    print()


#SARIMA
'''
The notation for the model involves specifying the order for the 
AR(p), I(d), and MA(q) models as parameters to an ARIMA function 
and AR(P), I(D), MA(Q) and m parameters at the seasonal level, e.g. 
SARIMA(p, d, q)(P, D, Q)m where “m” is the number of time steps in 
each season (the seasonal period)
'''

def sarima_model(feature):
    data = df1[feature]
    values = data.values
    values = values.astype('float32')
    split = int(len(list(df1["Dollar_eq"]))*0.8)
    #scaled = scale(values,0,1)
    train = values[:split]
    test = values[split:]
    #order=(1, 1, 1), seasonal_order=(1, 1, 1, 1)
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(train),len(train)+len(test )-1,typ="levels")
    x = []
    for i in range(len(test)):
        x.append((yhat[i],test[i]))
    #print(x[0:10])
    rmse = sqrt(mean_squared_error(test,yhat))
    print("\n\n SARIMA RMSE: %.5f" % rmse)
    print()


models = ['AR','MA','ARMA','ARIMA','SARIMA']
features = ["Dollar_eq","Volume","Percent_Change"]
'''
for i in features:
    print(i)
    ar_model(i)
    ma_model(i)
    arma_model(i)
    arima_model(i)
    #sarima_model(i)
'''


#Best univariate model = ARIMA

def get_arima_vals(feature):
    data = df1[feature]
    values = data.values
    values = values.astype('float32')
    split = int(len(list(df1["Dollar_eq"]))*0.8)
    #scaled = scale(values,0,1)
    train = values[:split]
    test = values[split:]
    #(p,d,q) = (1,1,1)
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(train),len(train)+len(test )-1,typ="levels")
    '''
    x=[]
    for i in range(len(test)):
        x.append((yhat[i],test[i]))
    print(x[0:10])
    rmse = sqrt(mean_squared_error(test,yhat))
    print("\n\n ARIMA RMSE: %.5f" % rmse)
    print()
    '''
    return yhat


#Forecasted features

f_doll = get_arima_vals("Dollar_eq")
f_vol = get_arima_vals("Volume")
f_percent = get_arima_vals("Percent_Change")
df_x = pd.DataFrame({"Dollar_eq":f_doll,"Volume":f_vol,"Percent_Change":f_percent})


'''
Simple Linear Regresion
'''

from sklearn import linear_model
from sklearn.metrics import r2_score

df1 = pd.DataFrame({"Dollar_eq":a,"Volume":b,"Percent_Change":c})

print("\n\nSIMPLE LINEAR REGRESSION\n\n")

split = int(len(list(df1["Dollar_eq"]))*0.8)
X_train = df1[:split].values
#X_test = df1[split:].values
#Using forecasted values instead of the values in the dataset
X_test = df_x.values

y = list(df["Price"])
y.reverse()

# Split the targets into training/testing sets
y_train = y[:split]
y_test = y[split:]

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


plt.plot(y_test)
plt.show()
plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()

'''
Ridge
'''

print("\n\nRIDGE REGRESSION\n\n")

'''
Here, aplha is a complexity parameter that controls the amount
 of shrinkage: the larger the value of alpha, the greater the 
 amount of shrinkage and thus the coefficients become more 
 robust to collinearity.
'''

regr = linear_model.Ridge (alpha = 10000)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()

'''
Rridge regression with built-in cross-validation
'''

print("\n\nRidge regression with built-in cross-validation\n\n")

regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 100000.0], cv=3)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()

'''
Lasso
'''

print("\n\nLASSO\n\n")

regr = linear_model.Lasso(alpha = 20)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()

'''
Elastic Net
'''
#alpha=0.1, l1_ratio=0.7

from sklearn.linear_model import ElasticNet

print("\n\nELASTIC NET\n\n")
enet = ElasticNet(alpha=100, l1_ratio=0.1)
y_pred = enet.fit(X_train, y_train).predict(X_test)
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()


'''
Least Angle Regression model 
'''

print("\n\nLeast Angle Regression model \n\n")

regr = linear_model.Lars(n_nonzero_coefs=2)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()


'''
LARS Lasso
'''

print("\n\nLARS Lasso\n\n")

regr = linear_model.LassoLars(alpha=0.5)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()


'''
Bayesian Ridge Regression
'''

print("\n\nBayesian Ridge Regression\n\n")

regr = linear_model.BayesianRidge()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()



'''
Automatic Relevance Determination Regression (ARD)
'''

print("\n\nAutomatic Relevance Determination Regression (ARD)\n\n")

from sklearn.linear_model import ARDRegression

regr = ARDRegression(compute_score=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()



'''
Passive Aggressive Regressor
'''


from sklearn.linear_model import PassiveAggressiveRegressor

print("\n\nPassive Aggressive Regressor\n\n")

regr = PassiveAggressiveRegressor(max_iter=100000, random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()


'''
Stochastic Gradient Descent
'''

print("\n\nStochastic Gradient Descent\n\n")

regr = linear_model.SGDRegressor(max_iter=10000)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()



'''
Theil-Sen Regressor
'''

print("\n\nTheil-Sen Regressor\n\n")

from sklearn.linear_model import TheilSenRegressor
regr = reg = TheilSenRegressor(random_state=0).fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()


'''
Huber Regressor
'''

print("\n\nHuber Regressor\n\n")

from sklearn.linear_model import HuberRegressor

regr = HuberRegressor(max_iter=10000, alpha=0.1).fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
plt.plot(y_pred)
plt.show()


'''
Polynomial Regression
'''

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

print("\n\nPolynomial Regression\n\n")

X_test = df_x.values
X_train = df1[:split].values

poly = PolynomialFeatures(degree=3)
X_train = poly.fit_transform(X_train)
#X_ = X_test
X_test = poly.fit_transform(X_test)

regr = LinearRegression()
regr.fit(X_train, y_train)


y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter([i for i in range(1,len(y_test)+1)],y_test)
#plt.scatter([i for i in range(1,len(y_test)+1)],y_pred)
plt.plot(y_pred)
plt.show()



'''
Polynomial Regression
'''

#print("\n\nPolynomial Regression\n\n")

X_train = df1[:split].values
#weights = np.polyfit(X_train, y_train, 3)














