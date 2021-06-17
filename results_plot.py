#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv('results.csv', index_col=0)


# In[3]:


rosen = df[[c for c in df.columns if 'rosen' in c]]
rosen.columns = [x.split('_')[-1] for x in rosen.columns]

rastrigin = df[[c for c in df.columns if 'rastrigin' in c]]
rastrigin.columns = [x.split('_')[-1] for x in rastrigin.columns]

ackley = df[[c for c in df.columns if 'ackley' in c]]
ackley.columns = [x.split('_')[-1] for x in ackley.columns]


# In[4]:


data = {
    'rosen' : rosen,
    'rastrigin': rastrigin,
    'ackley': ackley
}


# ### Results

# In[5]:


for f, d in data.items():
    print(f'Mean Results for the {f} function')
    print(d.mean().to_string(), end='\n\n')


# In[8]:


for f, d in data.items():
    bin_d = (d < 0.05).cumsum()
    fig = plt.figure(figsize=(13, 7))
    plt.plot(bin_d, label=bin_d.columns)
    plt.xlabel('# Iterations')
    plt.ylabel('cumulative # of solutions with distance < 5% of the target')
    plt.legend(fontsize='large')
    plt.title(f'Results for the {f} function')
    plt.savefig(f'imgs/{f}_line_plot', pad_inches=0)


# ### Plotting the results

# In[11]:


for f, d in data.items():
    fig = plt.figure(figsize=(13, 7))
    _ = plt.boxplot(d, labels=d.columns)
    plt.xlabel('Algorithm')
    plt.ylabel('Mean of the solutions')
    plt.title(f'Results for the {f} function')
    plt.savefig(f'imgs/{f}_box_plot', pad_inches=0)


# In[ ]:





# In[ ]:




