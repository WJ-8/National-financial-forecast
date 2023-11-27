import numpy as np


def GM11(x0):
    import numpy as np
    x1 = x0.cumsum()
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x0[1:].reshape((len(x0) - 1, 1))
    [[a], [u]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    f = lambda k: (x0[0] - u / a) * np.exp(-a * (k - 1)) - (x0[0] - u / a) * np.exp(-a * (k - 2))
    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) < 0.6745 * x0.std()).sum() / len(x0)
    return f, a, u, x0[0], C, P
