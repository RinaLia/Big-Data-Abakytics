#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


# In[21]:


penjualan= [7000000, 10000000, 7500000, 5000000, 17000000, 7000000, 14000000]
laba = [3000000, 4000000, 2000000, 1200000, 5000000, 2000000, 5000000]


# In[22]:


def linearRegresion(data):
    
    '''
        indeks[0] -> response variable -> x
        indeks[1] -> predictor variable -> y 
    '''
    x2=[]
    y2=[]
    xy=[]
    n = len(data[0])
    
    for x in data[0]:
        x2.append(x**2)
    
    for y in data[0]:
        y2.append(y**2)
    i=0;
    while(i<n):
        dump = data[0][i]*data[1][i]
        xy.append(dump)
        i+=1
    jmlhx = sum(data[0])
    jmlhy = sum(data[1])
    jmlhx2 = sum(x2)
    jmlhy2 = sum(y2)
    jmlhxy = sum(xy)
    a = ((jmlhy*jmlhx2)-(jmlhx*jmlhxy))/(n*jmlhx2-(jmlhx**2)) 
    b = ((n*jmlhxy)-(jmlhx*jmlhy))/(n*jmlhx2-(jmlhx**2))
    return(a,b)


# In[23]:


def gambarGrafik(dataProses):
    a,b = linearRegresion(dataProses)
    print("Nilai a adalah %.4f"%(a))
    print("Nilai b adalah %.4f"%(b))
    def f1(keanggotaan,a,b):
        hit = []
        for x in keanggotaan:
            y = b*x+a
            hit.append(y)
        return(hit)
    plt.scatter(dataProses[0],dataProses[1],label='data aktual',s=10)
    plt.plot(dataProses[0],f1(dataProses[0],a,b),c='k',label='hasil regresi',linewidth=0.5)
    plt.title("Hasil regresi Linear Sederhana")
    plt.ylabel("laba")
    plt.xlabel("penjualan")
    plt.legend()
    fig = plt.figure(1)
    plt.show()


# In[24]:


gambarGrafik([penjualan,laba])


# In[8]:


# Mengimpor library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sma
import statsmodels.formula.api as sm
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalizatio


# In[9]:


data = {'keuntungan' : [7000000, 10000000, 7500000, 5000000, 17000000, 7000000, 14000000],
        'jual' : [3000000, 4000000, 2000000, 1200000, 5000000, 2000000, 5000000],
        'bahanbaku':[100000, 200000, 300000, 400000, 500000, 600000, 700000]}
dataset = pd.DataFrame(data)
X = dataset.iloc[:, :].values

print(X,"\n") #untuk menampilkan variabel x, yaitu keuntungan dan bahan baku

y = dataset.iloc[:, 0].values #untuk menampilkan variabel y : keuntungan
print(y)


# In[10]:


# Membagi data menjadi the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Membuat model Multiple Linear Regression dari Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Memprediksi hasil Test set
y_pred = regressor.predict(X_test)
 
# Memilih model multiple regresi yang paling baik dengan metode backward propagation

X = sma.add_constant(X)
X_opt = X[:, [0, 1, 2]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 2]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [1,2]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

mpl.rcParams['legend.fontsize'] = 12


# In[11]:


fig = plt.figure() 
ax = fig.gca(projection ='3d') 
  
ax.scatter(X[:,0], X[:,1], X[:,2], label ='y', s = 5) 
ax.legend() 
ax.view_init(45, 0) 
  
plt.show()


# In[1]:


import numpy as np 
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt


# In[2]:


def generate_dataset(n): 
    x = [] 
    y = [] 
    random_x1 = np.random.rand() 
    random_x2 = np.random.rand() 
    for i in range(n): 
        x1 = i 
        x2 = i/2 + np.random.rand()*n 
        x.append([1, x1, x2]) 
        y.append(random_x1 * x1 + random_x2 * x2 + 1) 
    return np.array(x), np.array(y) 
  
x, y = generate_dataset(200)


# In[3]:


# create a linear model and fit it to the data
mpl.rcParams['legend.fontsize'] = 12
  
fig = plt.figure() 
ax = fig.gca(projection ='3d') 
  
ax.scatter(x[:, 1], x[:, 2], y, label ='y', s = 5) 
ax.legend() 
ax.view_init(45, 0) 
  
plt.show()


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


from sklearn.datasets import load_boston
boston_data = load_boston()
df =pd.DataFrame(boston_data.data,columns=boston_data.feature_names)


# In[9]:


df.head()


# In[10]:


X = df
y = boston_data.target
X_constant = sm.add_constant(X)
model = sm.OLS(y, X_constant)
lin_reg = model.fit()
lin_reg.summary()


# In[11]:


f_model = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
f_lin_reg = f_model.fit()
f_lin_reg.summary()


# In[12]:


print(lin_reg.predict(X_constant[:10]))
print(f_lin_reg.predict(X_constant[:10]))


# In[13]:


pd.options.display.float_format = '{:,.4f}'.format
corr = df.corr()
corr[np.abs(corr) < 0.65] = 0
plt.figure(figsize=(16,10))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()


# In[29]:


# yg digunakan: 
# Mengimpor library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sma
import statsmodels.formula.api as sm
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d, Axes3D

data = {'laba' : [3000000, 4000000, 2000000, 1200000, 5000000, 2000000, 5000000], 
        'penjualan' : [7000000, 10000000, 7500000, 5000000, 17000000, 7000000, 14000000],
        'bahanbaku': [459000, 657000, 455000, 355000, 699000, 512000, 658000 ]}

dataset = pd.DataFrame(data)
X = dataset.iloc[:, [1, 2]].values

print('penjualan (X1)\t bahanbaku(X2) ')
print(X,'\n') #untuk menampilkan variabel x, yaitu laba dan bahan baku

y = dataset.iloc[:, 0].values 
print('laba (Y): ')
print(y,'\n') #untuk menampilkan variabel y : laba
 
# Membagi data menjadi the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Membuat model Multiple Linear Regression dari Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
beta_topi= regressor.coef_
intersep = regressor.intercept_
b0=round(intersep, 2)
b1=round(beta_topi[0], 3)
b2=round(beta_topi[1], 3)
print('Didapatkan persamaan Y=',b0,'+',b1,'X1+',b2,'X2')

# Memprediksi hasil Test set
y_pred = regressor.predict(X_test)


# Memilih model multiple regresi yang paling baik dengan metode backward propagation

X = sma.add_constant(X)
X_opt = X[:, [0, 1, 2]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 2]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [1,2]]
regressor_OLS = sma.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

mpl.rcParams['legend.fontsize'] = 12
  
fig = plt.figure() 
ax = fig.gca(projection ='3d')
  
ax.scatter(X[:,0], X[:,1], X[:,2], label ='y', s = 5) 
ax.legend() 
ax.view_init(45, 0) 
  
plt.show()


# In[ ]:




