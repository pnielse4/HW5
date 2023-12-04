import numpy as np

def pfit(A):
    X1 =np.linspace(0, 1, 100)
    X2 = 1.0 - X1

    p1 = 10**(8.07131 - 1730.63/(20 + 233.426))
    p2 = 10**(7.43155 - 1554.679/(20 + 240.337))

    P = X1 * np.exp(A[0] * (A[1] * X2 / (A[0] * X1 + A[1] * X2)) ** 2) * p1 \
            + X2 * np.exp(A[1] * (A[0] * X1 / (A[0] * X1 + A[1] * X2)) ** 2) * p2

    return P

def loss(A):
    X1 =np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    P_actual = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])
    X2 = 1.0 - X1

    p1 = 10**(8.07131 - 1730.63/(20 + 233.426))
    p2 = 10**(7.43155 - 1554.679/(20 + 240.337))

    P = X1 * np.exp(A[0] * (A[1] * X2 / (A[0] * X1 + A[1] * X2)) ** 2) * p1 \
        + X2 * np.exp(A[1] * (A[0] * X1 / (A[0] * X1 + A[1] * X2)) ** 2) * p2

    l = P - P_actual
    total_l = np.dot(l, l)

    return total_l

def line_search(A):
    dl = 0.01
    aa = np.empty([100,2])
    grad = np.empty([100,1])
    theta = np.linspace(0, 2 * np.pi, 100)
    for n in np.linspace(0, 99, 100, dtype=int):
        aa[n, 0] = A[0] + np.cos(theta[n]) * dl
        aa[n, 1] = A[1] + np.sin(theta[n]) * dl
        grad[n] = (loss(aa[n, :]) - loss(A))/dl

    nmin = np.argmin(grad)
    grad_min = grad[nmin]
    new_a = aa[nmin, :]
    return new_a, grad_min



