import numpy as np
import matplotlib.pyplot as plt
import os
file_list = os.listdir('./log')
dir = './log/' + file_list[-1]
print(dir)
with open(dir,'r') as f:
    data = f.readlines()

log = []
for i in data:
    log.append(float(i.split(',')[1]))
plt.plot(log)
plt.xlabel('episode')
plt.ylabel('score per agent')
plt.grid()
plt.show()