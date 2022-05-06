import torch
import numpy as np
from collections import Counter
from channel import Channel
def myfun(data):
    tempdata = data
    tempdata[1] = 0
    return tempdata
a = torch.tensor([1,1,0,0,0,1,1,0,0,1])
index = torch.nonzero(a).squeeze()
n = len(index)
print(index,n)
a[index[-1]] = 0
print(a)
b = myfun(a.clone())
print(b,a)
c = b.tolist()
print(c)
print(c == set(c), c)
a = {1:0, 5:0, 6:1, 9:2}
min_set = min(a.values())
xxx = a.keys()
a = list(range(11))
a.remove(9)
print(a)


alive = torch.tensor([1,0,0,1,1])
alive = alive.view(1,1,5)
alive = alive.expand(1, 5, 5).unsqueeze(-1)
print(alive, alive.transpose(1,2))

# check for np.nonzero function
x = np.array([[1,0,1,1,1,0,0,1,0,1]])
print(x[np.nonzero(x)])
print(np.nonzero(x.squeeze()))

# check for non-alive, return an empty array 
a = np.array([0,0,0])
print(type(np.nonzero(a.squeeze())[0]))
print(type(np.nonzero(a.squeeze())[0].tolist()))

# check channel
alive = np.array([0,0,0,0,0,0])
mychannel = Channel()
a,b,c = mychannel.send(alive)
print(a)

# check empty list iter
a = [] 
for x in a:
    print('efeef')

# check loop return 
def loopfunction():
    x = 100
    while x >0:
        x -= 1
        return 100
print(loopfunction())