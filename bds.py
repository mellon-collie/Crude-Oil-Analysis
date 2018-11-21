from statsmodels.tsa import stattools
import pandas as pd


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


df = pd.read_csv("merge_oil.csv")

#BDS Test over columns
us_price = [conv_to_float(i) for i in list(df["US_Price"])]


print(stattools.bds(us_price, max_dim=2, epsilon=None, distance=1.5))


dollar_eq = [conv_to_float(i) for i in list(df["Dollar_eq"])]
print("Dollar_eq")
print(stattools.bds(dollar_eq, max_dim=2, epsilon=None, distance=1.5))

price = [float(i) for i in df["Price"]]
print("Price")
print(stattools.bds(price, max_dim=2, epsilon=None, distance=1.5))


Open = [float(i) for i in df["Open"]]
print("Open")
print(stattools.bds(Open, max_dim=2, epsilon=None, distance=1.5))

high = [float(i) for i in df["High"]]
print("High")
print(stattools.bds(high, max_dim=2, epsilon=None, distance=1.5))


low = [float(i) for i in df["Low"]]
print("Low")
print(stattools.bds(low, max_dim=2, epsilon=None, distance=1.5))



volume = [float(i) for i in df["Volume"]]
print("Volume")
print(stattools.bds(volume, max_dim=2, epsilon=None, distance=1.5))

p_change = [float(i) for i in df["Percent_Change"]]
print("Percent_Change")
print(stattools.bds(p_change, max_dim=2, epsilon=None, distance=1.5))


ioc_open = [conv_to_float(i) for i in list(df["IOC_Open"])]
print("IOC_Open")
print(stattools.bds(ioc_open, max_dim=2, epsilon=None, distance=1.5))

ioc_high = [conv_to_float(i) for i in list(df["IOC_High"])]
print("IOC_High")
print(stattools.bds(ioc_high, max_dim=2, epsilon=None, distance=1.5))

ioc_low = [conv_to_float(i) for i in list(df["IOC_Low"])]
print("IOC_Low")
print(stattools.bds(ioc_low, max_dim=2, epsilon=None, distance=1.5))

ioc_close = [conv_to_float(i) for i in list(df["IOC_Close"])]
print("IOC_Close")
print(stattools.bds(ioc_close, max_dim=2, epsilon=None, distance=1.5))

ioc_volume = [conv_to_float(i) for i in list(df["IOC_Volume"])]
print("IOC_Volume")
print(stattools.bds(ioc_volume, max_dim=2, epsilon=None, distance=1.5))


ongc_open = [conv_to_float(i) for i in list(df["ONGC_Open"])]
print("ONGC_Open")
print(stattools.bds(ongc_open, max_dim=2, epsilon=None, distance=1.5))

ongc_high = [conv_to_float(i) for i in list(df["ONGC_High"])]
print("ONGC_High")
print(stattools.bds(ongc_high, max_dim=2, epsilon=None, distance=1.5))

ongc_low = [conv_to_float(i) for i in list(df["ONGC_Low"])]
print("ONGC_Low")
print(stattools.bds(ongc_low, max_dim=2, epsilon=None, distance=1.5))

ongc_close = [conv_to_float(i) for i in list(df["ONGC_Close"])]
print("ONGC_Close")
print(stattools.bds(ongc_close, max_dim=2, epsilon=None, distance=1.5))

ongc_volume = [conv_to_float(i) for i in list(df["ONGC_Volume"])]
print("ONGC_Volume")
print(stattools.bds(ongc_volume, max_dim=2, epsilon=None, distance=1.5))

tci_open = [conv_to_float(i) for i in list(df["TCI_Open"])]
print("TCI_Open")
print(stattools.bds(tci_open, max_dim=2, epsilon=None, distance=1.5))

tci_high = [conv_to_float(i) for i in list(df["TCI_High"])]
print("TCI_High")
print(stattools.bds(tci_high, max_dim=2, epsilon=None, distance=1.5))

tci_low = [conv_to_float(i) for i in list(df["TCI_Low"])]
print("TCI_Low")
print(stattools.bds(tci_low, max_dim=2, epsilon=None, distance=1.5))

tci_close = [conv_to_float(i) for i in list(df["TCI_Close"])]
print("TCI_Close")
print(stattools.bds(tci_close, max_dim=2, epsilon=None, distance=1.5))

tci_volume = [conv_to_float(i) for i in list(df["TCI_Volume"])]
print("TCI_Volume")
print(stattools.bds(tci_volume, max_dim=2, epsilon=None, distance=1.5))
