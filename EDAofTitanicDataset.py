# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:23:29 2023

@author: anaya
"""

import pandas as pd

train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')

import matplotlib.pyplot as plt
import seaborn as sb

plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

train.shape
test.shape

train.isnull().sum()
test.isnull().sum()

sb.countplot(x='Survived', data=train)
plt.xticks([0,1],['not_Survived','Survived'])
plt.show()

train.groupby(['Sex', 'Survived'])['Survived'].count()

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
sb.countplot(x='Sex', hue='Survived', data=train,)
plt.xticks([0,1],['Male','Female'], rotation= 47)
plt.show()

sb.countplot(x='Pclass', hue='Survived', data = train)
plt.title('Pclass:Survived vs Dead')
plt.show()

pd.crosstab([train.Sex,train.Survived], train.Pclass,margins=True)


sb.catplot(x='Pclass',y='Survived', hue='Sex',kind='point', data=train)
plt.show()

print('Oldest people Survived was of:',train['Age'].max())
print('Youngest people Survived was of:',train['Age'].min())
print('Average people Survived was of:',train['Age'].mean())

f, ax=plt.subplots(1, 2, figsize=(18,8))
sb.violinplot(x='Pclass',y='Age',hue='Survived',data=train, split=True,ax=ax[0])
ax[0].set_title('PClass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sb.violinplot(x='Sex',y='Age', hue='Survived', data=train, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()

train['Initial'] = 0
for i in train:
    train['Initial'] = train['Initial'].str.extract('([A-Za-z]+)\.')

pd.crosstab(train.Initial, train.Sex)

train['Initial'] = train['Initial'].replace(
    ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don'],
    ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Miss', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr']
)

train.groupby('Initial')['Age'].mean()

train.loc[(train.Age.isnull()) & (train.Initial=='Mr'), 'Age']= 33
train.loc[(train.Age.isnull()) & (train.Initial=='Mrs'), 'Age']= 36
train.loc[(train.Age.isnull()) & (train.Initial=='Miss'), 'Age']= 22
train.loc[(train.Age.isnull()) & (train.Initial=='Master'), 'Age']= 5

train.Age.isnull().any()

f, ax= plt.subplots(1,2, figsize=(20,20))
train[train['Survived']==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black', color='red')
ax[0].set_title('Survived=0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['Survived']==1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black', color='green')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
ax[1].set_title('Survived=1')
plt.show()

sb.catplot(x='Pclass', y='Survived', col='Initial', data=train, kind='point')
plt.show()

pd.crosstab([train.SibSp], train.Survived)

f,ax=plt.subplots(1,2,figsize=(20,8))
sb.barplot(x='SibSp', y='Survived', data=train, ax=ax[0])
ax[0].set_title('SibSp vs Survived in barPlot')
sb.catplot(x='SibSp', y='Survived', data=train, ax=ax[1])
ax[1].set_title('SibSp vs Survived in FactorPlot')
plt.close(2)
plt.show()

f,ax=plt.subplots(1,2,figsize=(20,8))
sb.barplot(x='SibSp',y='Survived', data=train,ax=ax[0])
ax[0].set_title('SipSp vs Survived in BarPlot')
sb.catplot(x='SibSp',y='Survived', data=train,ax=ax[1])
ax[1].set_title('SibSp vs Survived in FactorPlot')
plt.close(2)
plt.show()

pd.crosstab(train.SibSp, train.Pclass)
