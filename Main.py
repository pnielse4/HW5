import matplotlib.pyplot as plt
from Functions import *
import numpy as np

A = np.array([1, 1])

X1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
P = [28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]



print(line_search(A))

grad = -20
aa_iter = A
while grad < float(-0.01):
    aa_iter, grad = line_search(aa_iter)

print(aa_iter)
np.linspace(0, 1, 100)
plt.plot(np.linspace(0, 1, 100), pfit(A), label='Initial guess')
plt.plot(np.linspace(0, 1, 100), pfit(aa_iter), label='A12=1.952 A21=1.694')
plt.scatter(X1, P, label='Data Points')
plt.ylabel('Pressure')
plt.xlabel('X1 water fraction')
plt.title('Curve fitting using Gradient Descent')
plt.legend()
plt.show()