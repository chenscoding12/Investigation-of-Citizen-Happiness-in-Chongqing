# Investigation-of-Citizen-Happiness-in-Chongqing

首先，代码使用了numpy、pandas、matplotlib、seaborn和statsmodels等常用数据处理和可视化库，这些库常被用于数据分析和机器学习等领域。

然后，代码使用pd.read_csv()读取了名为“telephone.csv”的数据文件，并使用np.nan将数据中的缺失值标记为NaN。接着，将数据的列名改为“depv”、“totcal”、“bnetp”和“snetp”。然后，根据这些列的值，生成了名为bnet和snet的新数组，并将其添加到名为telnum的新数据框中。然后，生成名为tel1的新数据框，并将其用于生成其他名为telnum2、telnum3、telnum5和telnum6的数据框。

接下来，代码使用tel.describe()和print(tel.depv.mean())分别计算并输出了数据集的描述性统计和depv列的平均值。

然后，代码使用matplotlib库和seaborn库绘制了多个图表，包括折线图、散点图、箱线图和QQ图等。这些图表用于可视化数据的不同特征和变量之间的关系。

最后，代码使用statsmodels库对数据进行了多次线性回归分析，并输出了结果的摘要统计信息和相关图表。回归模型包括使用不同的自变量组合。代码还使用了numpy库中的random.sample()函数，从数据集中随机抽取了800个样本。
