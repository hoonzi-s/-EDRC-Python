#데이터 시각화 기초

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,10)
y1 = 0.4*x**2
y2 = 4*x

#기본
plt.figure(1)
plt.plot(x,y1)
plt.show()

#그래프 중복해서 그리기
plt.figure(2)
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()

#중복된 그래프에서 legend 넣기
plt.figure(2)
plt.plot(x,y1, label='y1')
plt.plot(x,y2, label='y2')
plt.legend(loc='upper left')
plt.show()

#제목 넣기
plt.figure(1)
plt.plot(x,y1)
plt.title('Title insert example')
plt.show()

#축 값 넣기
plt.figure(1)
plt.plot(x,y1)
plt.title('Axis values insert example')
plt.xlabel('x value')
plt.ylabel('y value')
plt.show()

#subplot
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(x,y1)
plt.subplot(2,1,2)
plt.plot(x,y2)
plt.title('Subplot example')
plt.xlabel('x value')
plt.ylabel('y value')
plt.show()

#색 변경
plt.figure(3)
plt.subplot(2,2,1)
plt.plot(x,y1,'b')
plt.title('Color change example 1')

plt.subplot(2,2,2)
plt.plot(x,y1,'r')
plt.title('Color change example 2')
plt.xlabel('x value')
plt.ylabel('y value')

plt.subplot(2,2,3)
plt.plot(x,y2,'g')
plt.title('Color change example 3')

plt.subplot(2,2,4)
plt.plot(x,y2,'k')
plt.title('Color change example 4')
plt.show()

#선 모양 변경
plt.figure(4)
plt.subplot(1,3,1)
plt.plot(x,y1,'o')
plt.subplot(1,3,2)
plt.plot(x,y2,'--')
plt.subplot(1,3,3)
plt.plot(x,y2,'+r')
plt.title('Line change example')
plt.show()

#크기 변경
plt.figure(5, figsize=(3,6))
plt.plot(x,y1,'g--')
plt.show()
plt.figure(5, figsize=(3,3))
plt.plot(x,y1,'ro')
plt.title('Figure size change example')
plt.show()

#grid
plt.figure(5, figsize=(3,6))
plt.plot(x,y1,'g--')
plt.show()
plt.figure(5, figsize=(3,3))
plt.plot(x,y1,'ro')
plt.title('Figure size change example')
plt.grid()
plt.show()