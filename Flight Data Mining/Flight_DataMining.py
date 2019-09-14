#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('E:\Digitalent\Latihan progamming/flights.csv')
print(data)

#bar chart distance
data.groupby('dep_time')['arr_delay'].nunique().plot(kind='bar')
plt.show()

#pie chart distance
#data.groupby('carrier')['distance'].nunique().plot(kind='pie')
#plt.show()

#data.groupby('carrier')['distance'].nunique().plot(kind='pareto')
#plt.show()


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('E:\Digitalent\Latihan progamming/flights.csv')
print(data)

#bar chart distance
data.groupby('month')['arr_time'].nunique().plot(kind='bar')
plt.show()


# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('E:\Digitalent\Latihan progamming/flights.csv')
get_ipython().run_line_magic('matplotlib', 'inline')

#bar chart distance
cat_df_flights = df_flights.select_dtypes(include=['integer']).copy()
cat_df_flights.head()
#print(cat_df_flights['month'].value_counts())
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
month_count = cat_df_flights['month'].value_counts()
sns.set(style="darkgrid")
sns.barplot(month_count.index, month_count.values, alpha=0.9)
plt.title('Frequency Distribution of months')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.show()


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('E:\Digitalent\Latihan progamming/flights.csv')
get_ipython().run_line_magic('matplotlib', 'inline')

#bar chart distance
cat_df_flights = df_flights.select_dtypes(include=['integer']).copy()
cat_df_flights.head()
#print(cat_df_flights['month'].value_counts())
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
month_count = cat_df_flights['month'].value_counts()
sns.set(style="darkgrid")
sns.barplot(month_count.index, month_count.values, alpha=0.9)
plt.title('Frequency Distribution of months')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.show()


# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('E:\Digitalent\Latihan progamming/flights.csv')
get_ipython().run_line_magic('matplotlib', 'inline')

#bar chart distance
cat_df_flights = df_flights.select_dtypes(include=['integer']).copy()
cat_df_flights.head()
#print(cat_df_flights['month'].value_counts())
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
month_count = cat_df_flights['month'].value_counts()
sns.set(style="darkgrid")
sns.barplot(month_count.index, month_count.values, alpha=0.9)
plt.title('Frequency Distribution of months')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:




