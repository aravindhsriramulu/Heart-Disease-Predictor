#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Disease by using Neural Networks.
# 
# **About Dataset**
# 
# The dataset contains the following features:
# 
# - age(in years)
# - sex: (1 = male; 0 = female)
# - cp: chest pain type
# - trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# - chol: serum cholestoral in mg/dl
# - fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - restecg: resting electrocardiographic results
# - thalach: maximum heart rate achieved
# - exang: exercise induced angina (1 = yes; 0 = no)
# - oldpeak: ST depression induced by exercise relative to rest
# - slope: the slope of the peak exercise ST segment
# - ca: number of major vessels (0-3) colored by flourosopy
# - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# - target: 1 or 0
# 
# 
# ## **Content**
# 
# 1. [Data Visualization](#1.)
# 1. [Create Training and Testing Datasets](#2.)
# 1. [Building and Training the Neural Network](#3.)
# 1. [Improving Results - A Binary Classification Problem](#4.)
# 1. [Results and Metrics](#5.)
# 

# In[137]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras


# <a class="anchor" id="1."></a> 
# # 1. DATA VISUALIZATION (EDA)

# In[139]:


# read the csv
heart = pd.read_csv(r'C:\Aravindh/heart.csv')


# In[140]:


# print the shape of the DataFrame, so we can see how many examples we have
print( 'Shape of DataFrame: {}'.format(heart.shape))
print (heart.loc[1])


# In[141]:


# print the last twenty or so data points
heart.loc[280:]


# In[142]:


# remove missing data (indicated with a "?")
data = heart[~heart.isin(['?'])]
data.loc[280:]


# In[143]:


# drop rows with NaN values from DataFrame
data = data.dropna(axis=0)
data.loc[280:]


# In[144]:


# print the shape and data type of the dataframe
print(data.shape)
print(data.dtypes)


# In[145]:


# transform data to numeric to enable further analysis
data = data.apply(pd.to_numeric)
data.dtypes


# In[146]:


# print data characteristics, usings pandas built-in describe() function
data.describe()


# In[147]:


heart.loc[heart["sex"]==0,"sex"] = "Female"
heart.loc[heart["sex"]==1,"sex"] = "Male"

heart.loc[heart["cp"] == 0,"cp"] = "Typical Angina"
heart.loc[heart["cp"] == 1,"cp"] = "Atypical Angina"
heart.loc[heart["cp"] == 2,"cp"] = "Non Anginal Pain"
heart.loc[heart["cp"] == 3,"cp"] = "Asymptomatic"

heart.loc[heart["restecg"] == 0,"restecg"] = "Normal"
heart.loc[heart["restecg"] == 1,"restecg"] = "ST-T Wave Abnormality"
heart.loc[heart["restecg"] == 2,"restecg"] = "Left Ventricular Hypertrophy"

heart.loc[heart["slope"] == 0,"slope"] = "Unsloping"
heart.loc[heart["slope"] == 1,"slope"] = "Flat"
heart.loc[heart["slope"] == 2,"slope"] = "Downsloping"

heart.loc[heart["thal"] == 1,"thal"] = "Normal"
heart.loc[heart["thal"] == 2,"thal"] = "Fixed Defect"
heart.loc[heart["thal"] == 3,"thal"] = "Reversible Defect"

heart.loc[heart["fbs"] == 0,"fbs"] = "> 120mg/dL"
heart.loc[heart["fbs"] == 1,"fbs"] = "< 120mg/dL"

heart.loc[heart["exang"] == 0,"exang"] = "No"
heart.loc[heart["exang"] == 1,"exang"] = "Yes"

heart.loc[heart["target"] == 0,"target"] = "No heart disease found"
heart.loc[heart["target"] == 1,"target"] = "Has heart disease"


# In[148]:


heart.head()


# In[149]:


heart.tail()


# In[150]:


# --- Create List of Color Palletes ---
red_grad = ['#FF0000', '#BF0000', '#800000', '#400000', '#000000']


# ## Chest Pain Type (cp)

# In[151]:


colors=red_grad[0:4]
labels=['Type 0', 'Type 2', 'Type 1', 'Type 3']
order=heart['cp'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(16, 8))
plt.suptitle('Chest Pain Type Distribution', fontweight='heavy', fontsize=16, 
             fontfamily='sans-serif', color=red_grad[0])

# --- Pie Chart ---
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight='bold', fontsize=14,fontfamily='sans-serif', 
          color=red_grad[0])
plt.pie(heart['cp'].value_counts(), labels=labels, colors=colors, pctdistance=0.7, 
        autopct='%.2f%%', textprops={'fontsize':12},
        wedgeprops=dict(alpha=0.8, edgecolor=red_grad[1]))
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=red_grad[1])
plt.gcf().gca().add_artist(centre)

# --- Histogram ---
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', 
          color=red_grad[0])
ax = sns.countplot(x='cp', data=heart, palette=colors, order=order,
                   edgecolor=red_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=10,
             bbox=dict(facecolor='none', edgecolor=red_grad[0], linewidth=0.25,
                       boxstyle='round'))

plt.xlabel('Pain Type', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=red_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=red_grad[1])
plt.xticks([0, 1, 2, 3], labels)
plt.grid(axis='y', alpha=0.4)
countplt

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 30)
print('\033[1m'+'.: Chest Pain Type Total :.'+'\033[0m')
print('*' * 30)
heart.cp.value_counts(dropna=False)


# | Types | Chest Pain Type |	Criteria |
# | --- | --- | --- |
# | Type 0	 | Typical Angina  |	All criteria present |
# | Type 1	 | Atypical Angina |	2 of 3 criteria present |
# | Type 2  | Non Anginal Pain |	Less than one criteria present |
# | Type 3	 | Asymptomatic | None of criteria are satisfied |
# 
# -Angina: Discomfort when enough blood or oxygen is not circulated
# 
# -Non Anginal Pain: Pain in the chest usually caused by digestive tract
# 
# -Asymptomatic: No symptoms

# ## Resting ElectroCardiogram Results (restecg)

# In[152]:


# --- Setting Colors, Labels, Order ---
colors=red_grad[1:4]
labels=['1', '0', '2']
order=heart['restecg'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(16, 8))
plt.suptitle('Resting Electrocardiographic Distribution', fontweight='heavy', 
             fontsize=16, fontfamily='sans-serif', color=red_grad[0])

# --- Pie Chart ---
plt.subplot(1,2,1)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', 
          color=red_grad[0])
plt.pie(heart['restecg'].value_counts(), labels=labels, colors=colors, 
        wedgeprops=dict(alpha=0.8, edgecolor=red_grad[1]), autopct='%.2f%%',
        pctdistance=0.7, textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=red_grad[1])
plt.gcf().gca().add_artist(centre)

# --- Histogram ---
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', 
          color=red_grad[0])
ax = sns.countplot(x='restecg', data=heart, palette=colors, order=order,
                   edgecolor=red_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=10,
             bbox=dict(facecolor='none', edgecolor=red_grad[0], linewidth=0.25,
                       boxstyle='round'))

plt.xlabel('Resting Electrocardiographic', fontweight='bold', fontsize=11, 
           fontfamily='sans-serif', color=red_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=red_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 50)
print('\033[1m'+'.: Resting Electrocardiographic Results Total :.'+'\033[0m')
print('*' * 50)
heart.restecg.value_counts(dropna=False)


# | No. | Results |
# | --- | --- |
# | 0	  | Normal |
# | 1   |	Having ST-T wave abnormality |
# | 2	  | Showing probable or definite left ventricular hypertrophy by Estes' criteria |
# 
# Left Ventricular Hypertrophy: A heart's left pumping chamber has thickened and may not be pumping efficiently
# 
# ST-T wave abnormality: ST segment abnormality (elevation or depression) indicates myocardial ischaemia or infarction i.e. a heart attack

# ## Slope of peak exercise ST segment (slope)

# In[153]:


# --- Setting Colors, Labels, Order ---
colors=red_grad[2:5]
labels=['2', '1', '0']
order=heart['slope'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(16, 8))
plt.suptitle('Slope of the Peak Exercise Distribution', fontweight='heavy', 
             fontsize=16, fontfamily='sans-serif', color=red_grad[0])

# --- Pie Chart ---
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight='bold', fontsize=14,
          fontfamily='sans-serif', color=red_grad[0])
plt.pie(heart['slope'].value_counts(), labels=labels, colors=colors, 
        wedgeprops=dict(alpha=0.8, edgecolor=red_grad[1]), autopct='%.2f%%',
        pctdistance=0.7, textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=red_grad[1])
plt.gcf().gca().add_artist(centre)

# --- Histogram ---
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', 
          color=red_grad[0])
ax = sns.countplot(x='slope', data=heart, palette=colors, order=order,
                   edgecolor=red_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=10,
             bbox=dict(facecolor='none', edgecolor=red_grad[0], linewidth=0.25,
                       boxstyle='round'))

plt.xlabel('Slope', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=red_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=red_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 20)
print('\033[1m'+'.: Slope Total :.'+'\033[0m')
print('*' * 20)
heart.slope.value_counts(dropna=False)


# | No. | Slope | 
# | --- | --- | 
# | 1 | Upsloping | 
# | 2 | Flat | 
# | 3 | Downsloping | 
# 
# Horizontal or downsloping ST depression ≥ 0.5 mm at the J-point in ≥ 2 contiguous leads indicates myocardial ischaemia or blockage or arteries which eventually leads to heart disease

# ## Thallium Stress Test

# In[154]:


# --- Setting Colors, Labels, Order ---
colors=red_grad[0:4]
labels=['2', '3', '1', '0']
order=heart['thal'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(16,8))
plt.suptitle('"thal" Distribution', fontweight='heavy', fontsize=16, 
             fontfamily='sans-serif', color=red_grad[0])

# --- Pie Chart ---
plt.subplot(1,2,1)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', 
          color=red_grad[0])
plt.pie(heart['thal'].value_counts(), labels=labels, colors=colors, 
        wedgeprops=dict(alpha=0.8, edgecolor=red_grad[1]), 
        autopct='%.2f%%', pctdistance=0.7, textprops={'fontsize':12})

centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=red_grad[1])
plt.gcf().gca().add_artist(centre)

# --- Histogram ---
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', 
          color=red_grad[0])
ax = sns.countplot(x='thal', data=heart, palette=colors, order=order,
                   edgecolor=red_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=10,
             bbox=dict(facecolor='none', edgecolor=red_grad[0], linewidth=0.25,
                       boxstyle='round'))

plt.xlabel('Number of "thal"', fontweight='bold', fontsize=11, 
           fontfamily='sans-serif', color=red_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=red_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 20)
print('\033[1m'+'.: "thal" Total :.'+'\033[0m')
print('*' * 20)
heart.thal.value_counts(dropna=False)


# | No. | Results | Meaning |
# | --- | --- | --- |
# | 0 | Normal | Passed Thallium Test and condition is normal |
# | 1 | Fixed Defect | Heart tissue can't absorb thallium both under stress and in rest |
# | 2 | Reversible Defect | Heart tissue is unable to absorb thallium only under the exercise portion of the test |
# 
# A thallium stress test is a nuclear medicine study that shows your physician how well blood flows through your heart muscle while you're exercising or at rest and you're basically screwed if the result is a fixed defect or reversible defect. Fixed defect being worse

# In[155]:


print(heart["thal"].unique())

# replacing 0 - causes problems in pre processing
heart.loc[heart["thal"]==0,"thal"] = "Not taken the test"


# In[156]:


plt.figure(figsize=(10,8))

corr = heart.corr()

tick_labels = ["Age","Resting BP","Cholestrol","Max Heart Rate","Old Peak","Vessels colored"]

# Getting the Upper Triangle of the co-relation matrix
matrix = np.triu(corr)

# using the upper triangle matrix as mask 
corr_heatmap = sns.heatmap(corr,
            annot=True, 
            mask=matrix, 
            cmap="Reds",
            xticklabels=tick_labels,
            yticklabels=tick_labels
           )
plt.yticks(rotation=0)
plt.show()


# The correlation between age and blood pressure is almost strikingly same.
# 
# Age is positively correlated with almost all except heart rate. Increasing cholestrol, BP, Vessels colored from fluroscopy
# Maximum Heart Rate achieved by patient is negatively correlated with old peak (exercise relative to rest) and vessels colored

# In[157]:


### ST-T slopes for various Rest ECG results

colors = ["#BC544B","#FF0000"]
title_style = {
    "fontname":"monospace",
    "fontsize":25
}
plt.figure(figsize=(30,8))

slopes_st_t = heart.loc[heart["restecg"]=="ST-T Wave Abnormality"]
slopes_ventricular = heart.loc[heart["restecg"]=="Left Ventricular Hypertrophy"] 
slopes_normal = heart.loc[heart["restecg"]=="Normal"]


plt.subplot(1,3,1)
plt.title("ST-T Wave Abnormality",fontdict=title_style)
sns.countplot(x="slope",hue="sex",data=slopes_st_t,palette=colors)

plt.subplot(1,3,2)
plt.title("Ventricular Hypertrophy",fontdict=title_style)
sns.countplot(x="slope",hue="sex",data=slopes_ventricular,palette=colors)

plt.subplot(1,3,3)
plt.title("Normal",fontdict=title_style)
sns.countplot(x="slope",hue="sex",data=slopes_normal,palette=colors)

plt.show()


# Interestingly, there are no downsloping results when restecg results showed left ventricular hypertrophy. It is quite odd to note the fact that all of the patients having flat slope in ventricular hypertrophy are males and unsloping are females.

# In[158]:


plt.title("Count of males vs females")
sns.countplot(x="sex",data=heart,palette="Reds_r")
plt.show()


# As expected there are nearly 50 percent more males than females as patients.

# In[159]:


plt.title("FBS Count of Patients",fontdict={"fontname":"monospace","fontsize": 20})
sns.histplot(x="age",
             hue="fbs",
             data=heart,
             element="poly",
            )
plt.show()


# Most of the patients admitted have fasting blood sugar levels more than 120mg/L which means they are most probably diabetic and also most of the patients lie in the age group of 50-65.

# In[160]:


exercise_induced_angina = heart.loc[(heart["exang"]=="Yes") & (heart["thal"]!=0)]
age_unique=sorted(heart["age"].unique())

kwargs = dict(s=10)
plt.figure(figsize=(20,5))
# THALLIUM RESULTS
plt.subplot(1,2,1)
plt.title("Thallium Results",fontdict={"fontname":"monospace","fontsize":15})
sns.countplot(data=exercise_induced_angina,
              x="thal",
              palette="Reds"
             )

# FLUROSCOPY RESULTS
plt.subplot(1,2,2)
plt.title("Fluroscopy - Vessels colored",fontdict={"fontname":"monospace","fontsize":15})
sns.histplot(x="ca",hue="thal",data=exercise_induced_angina,palette="Reds_r",multiple="stack")
plt.show()


# MAX HEART RATE ACHIEVED
plt.figure(figsize=(20,5))
plt.title("Maximum Heart Rate achieved by all patients admitted",fontdict={"fontname":"monospace","fontsize":15})
plt.ylabel("Maximum heart rate")

# Calculating mean thalach else it would show blue patches of range
age_thalach_values=heart.groupby('age')['thalach'].count().values

mean_thalach = []
for i,age in enumerate(age_unique):
    mean_thalach.append(sum(heart[heart['age']==age].thalach)/age_thalach_values[i])

sns.pointplot(x=age_unique,y=mean_thalach,markers=['o'],scale=0.5,color="red")

plt.show()


# Most of the people who had exercise induced angina showed reversible defect in thallium test which is true as reversible defect is when heart tissue is unable to absorb thallium only under the exercise portion of the test, followed by fixed defect which means under stress and rest and very small percentage of patients got the result of normal
# 
# More vessels colored implies blood flow is proper and hence lesser count of admitted patients. Fixed defect thallium test is more visible in patients with number of colored vessels as 0
# 
# Average thalach (Maximum heart rate achieved) is decreasing with age 

# In[161]:


# Cholestrol is in mg/dl according to this data and also dataset hence we can check the range and determine heart disease or not
kwargs=dict(s=30)
fig, ax = plt.subplots(nrows=1,figsize=(10,8))
plt.title("Cholestrol Levels vs Heart Disease",fontdict={"fontname":"monospace","fontsize": 20})
sns.scatterplot(x="age",
            y="chol",
            hue="target",
            data=heart,
            palette="Reds",
            **kwargs)

ax.axhspan(0,200,alpha=0.2,color='green')
ax.axhspan(200,239,alpha=0.2,color='yellow')
ax.axhspan(239,600,alpha=0.2,color='red')

plt.show()


# Most of the Patients having cholestrol above and equal to the ideal range were having more chances in suffering from heart disease. However, the patients having heart disease and vice versa is quite spread out.
# 
# Surprisingly, patients whose cholestrol level touched the mark of 400 still did not succumb to a heart disease. On the flip side, patients whose cholestrol levels were in the range of 160-180 fell to heart disease. The most concentrated age range for the possibility of having a heart disease is 41, 44, 51, 52 and 53

# In[162]:


plt.title("Patient's age diagnosed with heart disease",fontdict={"fontname":"monospace","fontsize": 15})
have_heart_disease = heart.loc[heart["target"]=="Has heart disease"]
sns.swarmplot(x="age",data=have_heart_disease, color='red')
plt.show()


# Heart diseases are unbiased to age. Affects everyone how generous

# In[163]:


plt.figure(figsize=(10,5))
plt.title("Chest Pain and Heart Disease",fontdict={"fontname":"monospace","fontsize": 20})
sns.histplot(x="cp",
             data=heart,
             hue="target",
             multiple="stack",
             palette="Reds_r")
plt.show()


# Most of the patients suffering non-anginal and atypical anginal chest pain have a higher risk of acquiring a heart disease. An equal proportion of people suffering from asymptomatic chest pain had a heart disease and most of the patients admitted having typical angina did not have a heart disease.

# <a class="anchor" id="2."></a> 
# # 2.Create Training and Testing Datasets
# 
# Now that we have preprocessed the data appropriately, we can split it into training and testings datasets. We will use Sklearn's train_test_split() function to generate a training dataset (80 percent of the total data) and testing dataset (20 percent of the total data). 
# 

# In[164]:


X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])


# In[165]:


X[0]


# In[166]:


mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std


# In[167]:


X[0]


# In[168]:


# create X and Y datasets for training
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)


# In[169]:


# convert the data to categorical labels
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])


# In[170]:


X_train[0]


# <a class="anchor" id="3."></a> 
# # 3.Building and Training the Neural Network
# 
# We can begin building a neural network to solve this classification problem. Using keras, we will define a simple neural network with one hidden layer. Since this is a categorical classification problem, we will use a softmax activation function in the final layer of our network and a categorical_crossentropy loss during our training phase.

# In[171]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())


# In[172]:


# fit the model to the training data
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=50, batch_size=10)


# In[173]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[174]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# <a class="anchor" id="4."></a> 
# # 4.Improving Results - A Binary Classification Problem

# In[175]:


# convert into binary classification problem - heart disease or no heart disease
Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# In[176]:


# define a new keras model for binary classification
def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())


# In[177]:


# fit the binary model on the training data
history=binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=50, batch_size=10)


# In[178]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[179]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# <a class="anchor" id="5."></a> 
# # 5.Results and Metrics

# In[180]:


# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model.predict(X_test), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))


# In[181]:


import tensorflow as tf
tf.keras.models.save_model(binary_model,'my_model2.hdf5')


# # Streamlit Webapp

# In[182]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart disease Prediction App
This app predicts If a patient has a heart disease
Data obtained from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset.
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')

    sex  = st.sidebar.selectbox('Sex',(0,1))
    cp = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    res = st.sidebar.number_input('Resting electrocardiographic results: ')
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina: ',(0,1))
    old = st.sidebar.number_input('oldpeak ')
    slope = st.sidebar.number_input('he slope of the peak exercise ST segmen: ')
    ca = st.sidebar.selectbox('number of major vessels',(0,1,2,3))
    thal = st.sidebar.selectbox('thal',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = data.copy()
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df,heart_dataset],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
#df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)
# Reads in saved classification model
load_clf = tf.keras.models.load_model('my_model2.hdf5')

# Apply model to make predictions
prediction = load_clf.predict(df)

st.subheader('Prediction')
st.write(prediction)


# In[ ]:




