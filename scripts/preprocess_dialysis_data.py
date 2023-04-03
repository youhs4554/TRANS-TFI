#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
pd.set_option('display.max_columns', None)


# In[3]:


raw_file = '../data/dialysis/PatientData.xlsx'

data = pd.read_excel(raw_file, 'PatientData', index_col=0, engine='openpyxl').reset_index()
meta = pd.read_excel(raw_file, 'META', index_col=0, engine='openpyxl', keep_default_na=False).reset_index()
mapping = pd.read_excel(raw_file, 'MAPPING', index_col=0, engine='openpyxl').reset_index()


# In[4]:


MISSING_TOLERANCE = 0.1


# In[5]:


print(f"Raw data : {data.shape}")


# In[6]:


# 사용되는 컬럼정보 추출
meta_info = meta[(meta['KEY'] == 1) | (meta['INPUT'] == 1) | (meta['LABEL'] == 1)]
columns_info = meta_info['COL_NM'].values
print(f"Select {len(columns_info)} among {data.shape[1]} features")


# In[7]:


# 가용 컬럼리스트에 따른 데이터
data_info = data[columns_info]
print(f"Selected data : {data_info.shape}")


# In[8]:


def report_stats(df):
    print(f"Dataset size : {df.shape}")
    display(df.head())
    df.info(verbose=True, show_counts=True)
    


# In[9]:


report_stats(data_info)


# In[10]:


import numpy as np

# 엑셀에 있는 MAPPING에 따라 컬럼별로 매핑정보를 가져온다.
def mapping_dict(col_nm):
    mapping_temp = mapping[mapping['COL_NM'] == col_nm]
    
    map_dict = {}
    for i, row in mapping_temp.iterrows():
        val_from = str(row['VAL_FROM'])
        
        for val in val_from.split(','):
            if '~' in val:
                val_range = val.split('~')
                for subval in range(int(val_range[0]), int(val_range[1]) + 1):
                    map_dict[str(subval)] = str(row['VAL_TO'])
#                 print(val_range, int(val_range[0]), int(val_range[1]) + 1)
            else:
                map_dict[str(val)] = str(row['VAL_TO'])
        
    return map_dict
        
    
def mapping_value(x, map_dict):
    
    try:
        ret = map_dict.get(str(int(x)), np.nan)
    except ValueError as e:
        #print(e)
        ret = np.nan
    
    return ret

# 매핑 정보에 따라 데이터 매핑해서 data_info 저장
for map_col in mapping['COL_NM'].unique():
    map_dict = mapping_dict(map_col)
    data_info[map_col] = data_info[map_col].map(lambda x: mapping_value(x, map_dict))


# In[11]:


report_stats(data_info)


# In[12]:


# 결측치로 처리될 문자열 리스트
strings_missing = ['ND', 'nodata', 'Not Answer', 'Unknown']

# missing data 이면 None 으로 일관성있게 처리
def replace_missing_data_to_empty(x):
    if x in strings_missing:
        return np.nan
    else:
        return x
    
for ix, row in meta_info.iterrows():
    col_nm = row['COL_NM']
    col_type = row['TYPE']
    
    data_col = data_info[col_nm]
    data_col_dtype = row.TYPE
    
    # 결측치로 처리될 문자열 리스트가 존재하면 빈값처리
    if len(set(strings_missing).intersection(data_col.unique())) > 0:
        data_info[col_nm] = data_col.map(lambda x: replace_missing_data_to_empty(x))


# In[13]:


report_stats(data_info)


# In[14]:


data_info.apply(lambda col: col.isin(strings_missing).any(), axis=1).any()


# In[15]:


data_info.set_index("ID", inplace=True)
data_info.reset_index(drop=True, inplace=True)


# In[16]:


data_info.dropna(axis=1, thresh=len(data_info)*(1-MISSING_TOLERANCE), inplace=True)
report_stats(data_info)


# In[17]:


print("Missing rate for each column (sorted):")

for name,missing_rate in data_info.isnull().mean().sort_values(ascending=False).items():
    print(name, missing_rate)


# In[18]:


# drop label columns
label_cols = ["death_1yr", "death_3yr", "death_5yr"]
data_info = data_info.drop(label_cols, axis=1)


# In[19]:


data_info.head()


# In[20]:


for i, col in data_info.items():
    data_info[i] = pd.to_numeric(col, errors="ignore")


# In[21]:


data_info.describe()


# In[22]:


numeric_columns = data_info.describe().columns.tolist()
categorical_columns = list(set(data_info)-set(numeric_columns))


# In[23]:


for col in categorical_columns:
    print(data_info[col].value_counts())


# In[24]:


data_info[numeric_columns].info()


# In[25]:


labels = data.loc[data_info.index, ["death", "survival_days"]+label_cols]
labels


# In[26]:


data_info


# In[27]:


def report_label_stats(labels):
    for death in [0, 1]:
        print(f"death : {death}")
        display(labels \
        .query(f"death == {death}")[label_cols] \
        .astype(str).apply(lambda row: "/".join(row), axis=1).value_counts())


# In[28]:


report_label_stats(labels)


# In[29]:


get_ipython().system('pip install plotly')


# In[30]:


import plotly.graph_objects as go

pure_survivor_grp = labels.query("death_1yr==0 & death_3yr==0 & death_5yr==0 & death==0")
conditional_survivor_grp = labels.query("death_1yr==0 & death_3yr==0 & death_5yr==0 & death==1")

fig = go.Figure()
fig.add_trace(go.Histogram(x=pure_survivor_grp.survival_days, name="fully survive"))
fig.add_trace(go.Histogram(x=conditional_survivor_grp.survival_days, name="partially survive(die 5yr later)"))
fig.update_layout(barmode='stack')
fig.update_traces(opacity=0.75)
fig.show()


# In[31]:


import plotly.express as px

fig = px.histogram(labels, x='survival_days', color='death')
fig.update_layout(barmode='stack')
fig.show()


# In[32]:


full_dataset = pd.concat([data_info, labels], axis=1)
full_dataset


# In[33]:


report_stats(full_dataset)


# In[34]:


full_dataset.to_excel("../data/dialysis/prepared_full.xlsx", index=False)


# In[35]:


full_dataset.shape


# In[36]:


full_dataset


# In[37]:


import json

column_types = {
    "num": numeric_columns,
    "cat": categorical_columns
}

json.dump(column_types, open("../data/dialysis/column_types.json", "w"))


# In[38]:


pd.read_excel("../data/dialysis/prepared_full.xlsx")


# In[ ]:




