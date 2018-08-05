import numpy as np
import pylab
import matplotlib.pyplot as plot
import scipy.stats as stats
# 拟合直线
data=np.mat([[1,200,105,3,False],
             [2,165,80,2,False],
             [3,184.5,120,2,False],
             [4,116,70.8,1,False],
             [5,270,150,4,True]])
coll=[]
for row in data:
    print(row)
    coll.append(row[0,2])
stats.probplot(coll,plot=pylab,rvalue=True)
#pylab.show()
print(coll)
#坐标图的展示
import pandas as pd
rocksVMines=pd.DataFrame([[1,200,105,3,False],
                         [2,165,80,2,False],
                         [3,184.5,120,2,False],
                         [4,116,70.8,1,False],
                         [5,270,150,4,True]
    ])
dataRows1=rocksVMines.iloc[1,0:3]
print(dataRows1)
datarows2=rocksVMines.iloc[2,0:3]
print(datarows2)
plot.scatter(dataRows1,datarows2)
plot.xlabel("Attribute2")
plot.ylabel(("Attribute3"))
#plot.show()

datarows3=rocksVMines.iloc[3,0:3]
print(datarows3)
plot.scatter(datarows2,datarows3)
plot.xlabel("Attribute2")
plot.ylabel("Attribute3")
#plot.show()

# 数据集合的展示
import pandas as pd			
import os								
import matplotlib.pyplot as plot									
filePath = (r"E:\TensorFlowDeeperLearningcode\data\dataTest.csv")									
dataFile = pd.read_csv(filePath,header=None, prefix="V")				

target = []													
for i in range(200):											
    if dataFile.iat[i,10] >= 7:									
        target.append(1.0 + uniform(-0.3, 0.3))					
    else:
        target.append(0.0 + uniform(-0.3, 0.3))					
dataRow = dataFile.iloc[0:200,10]								
plot.scatter(dataRow, target, alpha=0.5, s=100)						
plot.xlabel("Attribute")										
plot.ylabel("Target")											
plot.show()	