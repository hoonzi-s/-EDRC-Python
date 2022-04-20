import numpy as np
import matplotlib.pyplot as plt

#raw data
np.random.seed(0)
data = 10*np.random.rand(20)
print(data)

#outlier data 생성
data[10]=100
data[15]=70

#Figure
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(data,'bo')
plt.xlabel('index')
plt.ylabel('data')
plt.title('Outlier data example')
ax.annotate('Outlier data (1)', xy=(10,100), xytext=(10,85), arrowprops=dict(facecolor='black'))
ax.annotate('Outlier data (2)', xy=(15,70), xytext=(15,55), arrowprops=dict(facecolor='black'))
plt.show()

#Boxplot
plt.boxplot(data)
plt.title('Boxplot of data example')
plt.xlabel('data')
plt.ylabel('data value')
plt.show()
