# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:25:24 2018

@author: chens
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D


tel = pd.read_csv('telephone.csv', encoding = 'GBK')
tel == np.nan
tel.columns = ['depv', 'totcal', 'bnetp', 'snetp']
bnet = np.array(tel.totcal*tel.bnetp)
snet = np.array(tel.totcal*tel.snetp*tel.bnetp)
telnum = pd.DataFrame({'bnet':bnet,
                       'snet':snet,
                       'tolcal':tel.totcal,
                       'depv':tel.depv})
tel1 = pd.DataFrame({'bnetp':tel.bnetp,
                     'snetp':tel.snetp,
                        'totcal':tel.totcal})
telnum2 = pd.DataFrame({'diffbs':np.array(telnum.bnet+telnum.snet-tel.totcal)})
telnum3 = pd.DataFrame({'actcal':np.array(tel.totcal*tel.bnetp*tel.snetp)})
telnum5 = pd.DataFrame({'hdv':np.array(tel.totcal*(tel.bnetp-tel.snetp))})
telnum6 = pd.DataFrame({'bnetp':tel.bnetp,
                        'totcal':tel.totcal})

tel.describe()
print(tel.depv.mean())

fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(tel.depv[1:100], '--', label = 'Dependent variable')
ax.plot(tel.totcal[1:100], '--', label = 'Total call volume')
ax.plot(tel.bnetp[1:100], '--', label = 'The ratio of big network')
ax.plot(np.array(tel.snetp)[1:100], '--', label = 'The ratio of small network')
ax.legend(loc='best')

fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(telnum.depv[1:50], '--', label = 'The difference of profit')
ax.plot((tel.totcal[1:50]), '--', label = 'Total call volume')
ax.plot((telnum.bnet[1:50]), '--', label = 'Big network')
ax.plot(telnum.snet[1:50], '--', label = 'Small network')
ax.legend(loc='best')

pd.DataFrame({'bnet-snet':np.array(bnet-snet),
              'depv':tel.depv})[1:200].plot()

fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(tel.totcal[1:100], label = 'Total call volume')
ax.plot((telnum.bnet[1:100]+telnum.snet[1:100]), label = 'Sum of several factors')
ax.plot(tel.depv[1:100], label = 'Dependent variable')
ax.legend(loc='best')

telnum[1:50].plot()
pd.DataFrame(np.array(tel.bnetp+tel.snetp)[1:100]).plot()
pd.DataFrame({'depv':tel.depv,
              'actcal':telnum3.actcal})[1:200].plot()
pd.DataFrame({'depv':tel.depv,
              'snet':tel.snetp})[1:50].plot()

fig = plt.figure()
fig.add_subplot(1, 2, 1)
sns.set(font_scale=2)
sns.boxplot(tel)
tel.corr = tel.corr()
fig.add_subplot(1, 2, 2)
sns.heatmap(tel.corr)


telnum_model = sm.add_constant(telnum)
tel_model1 = sm.add_constant(tel1)
telnum_model2 = sm.add_constant(telnum2)
telnum_model3 = sm.add_constant(telnum3)
telnum_model4 = sm.add_constant(tel.snetp)
telnum_model6 = sm.add_constant(telnum6)
reg = sm.OLS(tel.depv, tel_model1)
results = reg.fit()
print(results.summary())

telnum2[100:500].plot()

fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(tel.depv[1:100],'o-', label = 'Data')
ax.plot(results.fittedvalues[1:100], 'k--', label = 'OLS')
ax.legend(loc='best')

outliers = results.get_influence()
leverage = outliers.hat_matrix_diag
dffits = outliers.dffits[0]
cook = outliers.cooks_distance[0]
resid_stu = outliers.resid_studentized_external
covratio = outliers.cov_ratio
contat1 = pd.concat([pd.Series(leverage, name = 'leverage'),pd.Series(dffits, name = 'dffits'),
                     pd.Series(resid_stu, name = 'resid_stu'),pd.Series(cook, name = 'cook'),
                     pd.Series(covratio, name = 'covratio'),],axis = 1)
tel_outliers = pd.concat([tel,contat1], axis = 1)
tel_outliers.head()

outliers_ratio = sum(np.where((np.abs(tel_outliers.resid_stu)>2),1,0))/tel_outliers.shape[0]
outliers_ratio

tel_outliers = tel_outliers.loc[np.abs(tel_outliers.resid_stu)<=2,]

telhd1 = pd.DataFrame({'bnetp':tel_outliers.bnetp,
                      'snetp':tel_outliers.snetp,
                      'totcal':tel_outliers.totcal})
telhd1_model = sm.add_constant(telhd1)
reg1 = sm.OLS(tel_outliers.depv, telhd1_model)
results1 = reg1.fit()
print(results1.summary())

telhd2 = pd.DataFrame({'bnetp':tel_outliers.bnetp,
                      'totcal':tel_outliers.totcal})
telhd2_model = sm.add_constant(telhd2)
reg2 = sm.OLS(tel_outliers[1:800].depv, telhd2_model[1:800])
results2 = reg2.fit()
print(results2.summary())
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(tel_outliers.depv.reset_index(drop=True)[800:1000],'o-', label = 'Data')
datapre = pd.DataFrame({'bnetp':tel_outliers.bnetp,
                        'totcal':tel_outliers.totcal})
datapre_model = sm.add_constant(datapre)
ax.plot(pd.DataFrame(results2.predict(datapre_model)).reset_index(drop=True)[800:1000], 'k--', label = 'OLS')
ax.legend(loc='best')

test = pd.DataFrame({'test':tel_outliers.depv.reset_index(drop=True)[800:1000]})
pred = pd.DataFrame({'pred':results2.predict(datapre_model)}).reset_index(drop=True)[800:1000]
pred_model = sm.add_constant(pred)
reg_t = sm.OLS(test, pred)
res_t = reg_t.fit()
print(res_t.summary())

x,y,z = tel_outliers.depv, tel_outliers.totcal, tel_outliers.bnetp
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()
merge = pd.DataFrame({'pred':pred.pred,
                      'test':test.test})
fig = plt.figure()
fig.add_subplot(1, 1, 1)
sns.set(font_scale=2)
sns.boxplot(merge)

import statsmodels.api as sm
import pylab
sm.qqplot(results2.predict(datapre_model)[1:500], line='s')
pylab.show()

import random as rd
slice = rd.sample(as.list(tel), 800)
