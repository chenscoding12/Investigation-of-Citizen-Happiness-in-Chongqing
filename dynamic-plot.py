# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:34:21 2018

@author: chens
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import statsmodels.api as sm
                   
tel = pd.read_csv('telephone.csv', encoding = 'GBK')
tel == np.nan
tel.columns = ['depv', 'totcal', 'bnetp', 'snetp']
bnet = np.array(tel.totcal*tel.bnetp)
snet = np.array(tel.totcal*tel.snetp*tel.bnetp)
telnum = pd.DataFrame({'bnet':bnet,
                       'snet':snet,
                       'tolcal':tel.totcal,
                       'depv':tel.depv})
telnum1 = pd.DataFrame({'bnetp':tel.bnetp,
                        'snetp':tel.snetp,
                        'totcal':tel.totcal})
telnum2 = pd.DataFrame({'diffbs':np.array(telnum.bnet+telnum.snet-tel.totcal)})
telnum3 = pd.DataFrame({'actcal':np.array(tel.totcal*tel.bnetp*tel.snetp)})
telnum5 = pd.DataFrame({'hdv':np.array(tel.totcal*(tel.bnetp-tel.snetp))})
telnum6 = pd.DataFrame({'bnetp':tel.bnetp,
                        'totcal':tel.totcal})

tel.describe()
print(tel.depv.mean())


telnum_model = sm.add_constant(telnum)
telnum_model1 = sm.add_constant(telnum1)
telnum_model2 = sm.add_constant(telnum2)
telnum_model3 = sm.add_constant(telnum3)
telnum_model4 = sm.add_constant(tel.snetp)
telnum_model6 = sm.add_constant(telnum6)
reg = sm.OLS(tel.depv, telnum_model1)
results = reg.fit()
print(results.summary())





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
datapre = pd.DataFrame({'bnetp':tel_outliers.bnetp,
                        'totcal':tel_outliers.totcal})     
datapre_model = sm.add_constant(datapre)

def simple_plot():
    """
    simple plot
    """
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=100)

    telhd2 = pd.DataFrame({'bnetp':tel_outliers.bnetp,
                           'totcal':tel_outliers.totcal})
    telhd2_model = sm.add_constant(telhd2)
    reg2 = sm.OLS(tel_outliers[1:800].depv, telhd2_model[1:800])
    results2 = reg2.fit()
    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(102):
        plt.cla()
        # 设定标题等
        plt.title("Prediction(Index = 900 ~ 1000)")
        plt.grid(True)

        # 画两条曲线
        plt.plot(tel_outliers.depv.reset_index(drop=True)[900:1000],'o--', label = 'Data')
        plt.plot(pd.DataFrame(results2.predict(datapre_model)).reset_index(drop=True)[900:900+index], 'r-', label = 'OLS')

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper left", shadow=True)

        # 暂停
        plt.pause(0.01)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return
# simple_plot()

simple_plot()
