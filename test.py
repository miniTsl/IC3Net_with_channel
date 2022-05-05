import torch
import numpy as np
from collections import Counter
def myfun(data):
    data[1] = 0
    return data
a = torch.tensor([1,1,0,0,0,1,1,0,0,1])
index = torch.nonzero(a).squeeze()
n = len(index)
print(index,n)
a[index[-1]] = 0
print(a)
b = myfun(a)
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