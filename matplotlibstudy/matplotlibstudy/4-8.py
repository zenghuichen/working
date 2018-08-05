import pandas as pd			
import os								
import matplotlib.pyplot as plot	
import random
filePath = (r"E:\TensorFlowDeeperLearningcode\data\dataTest.csv")									
dataFile = pd.read_csv(filePath,header=None, prefix="V")				

target = []													
for i in range(200):											
    if dataFile.iat[i,10] >= 7:									
        target.append(1.0 + random.uniform(-0.3, 0.3))					
    else:
        target.append(0.0 + random.uniform(-0.3, 0.3))					
dataRow = dataFile.iloc[0:200,10]	

plot.scatter(dataRow, target, alpha=0.5, s=100)						
plot.xlabel("Attribute")										
plot.ylabel("Target")											
plot.show()	
