#!/usr/bin/env python
# coding: utf-8

# In[24]:



import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

import seaborn as sns



# for market basket analysis
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\gmc\Market_Basket_Optimisation.csv', header = None)
df


# In[25]:


data.describe()


# In[26]:


# making each customers shopping items an identical list
trans = []
for i in range(0, 7501):
    trans.append([str(data.values[i,j]) for j in range(0, 20)])

# conveting it into an numpy array
trans = np.array(trans)

# checking the shape of the array
print(trans.shape)


# In[27]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
data = te.fit_transform(trans)
data = pd.DataFrame(data, columns = te.columns_)

# getting the shape of the data
data.shape


# In[28]:


data.columns


# In[29]:


data.head()


# In[30]:


from mlxtend.frequent_patterns import apriori

#Now, let us return the items and itemsets with at least 5% support:
apriori(data, min_support = 0.01, use_colnames = True)


# In[31]:


frequent_itemsets = apriori(data, min_support = 0.05, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[32]:



frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.01) ]


# In[33]:


frequent_itemsets[ (frequent_itemsets['length'] == 1) &
                   (frequent_itemsets['support'] >= 0.01) ]


# In[ ]:




