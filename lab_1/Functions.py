from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def hessian(self, x):
        pass

# обертка для подсчета вызовов фукнции для тестов
class CallCounter:
    def __init__(self, target_function):
        self.func = target_function
        self.calls = 0
        self.grad_calls = 0
        self.hessian_calls = 0
        self.summary = 0

    def __call__(self, x):
        self.calls += 1
        self.summary += 1
        return self.func(x)

    def gradient(self, x):
        self.grad_calls += 1
        self.summary += 1
        return self.func.gradient(x)

    def hessian(self, x):
            self.hes_calls += 1
            self.summary += 1
            return self.func.hessian(x)

class QuadraticFunction(Function):
    def __init__(self, A, b=None, c=0):
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A should be square.")

        if not np.allclose(A, A.T):
            raise ValueError("Matrix A should be symmetric.")

        if b is None:
            b = np.zeros(A.shape[0])
        
        self.A = A
        self.n = A.shape[0]
        self.b = b
        self.c = c

    def __call__(self, x):
        x = np.asarray(x)

        return 0.5 * x @ self.A @ x + self.b @ x + self.c

    def gradient(self, x):
        x = np.asarray(x)
        
        return self.A @ x + self.b

    def hessian(self, x=None):
        return self.A

class RosenbrockFunction(Function):
    def __call__(self, x):
        x = np.asarray(x)

        if x.shape[0] != 3:
            raise ValueError("Input should be a vector of 3 values.")

        x1, x2, x3 = x

        return (1 - x1)**2 + 100 * (x2 - x1**2)**2 + (1 - x2)**2 + 100 * (x3 - x2**2)**2

    def gradient(self, x):
        x1, x2, x3 = x
        
        df_dx1 = 2 * x1 - 2 + 100 * (-4 * x2 * x1 + 4 * x1**3)

        df_dx2 = 100 * (2 * x2 - 2 * x1**2) - 2 + 2 * x2 + 100 * (-4 * x3 * x2 + 4 * x2**3)

        df_dx3 = 100 * (2 * x3 - 2 * x2**2)

        return np.array([df_dx1, df_dx2, df_dx3])

    def hessian(self, x):
        x1, x2, x3 = x
        
        return np.array([
            [
                1200 * x1**2 - 400 * x2 +2,
                -400 * x1,
                0
            ],
            [
                -400 * x1,
                1200 * x2**2 - 400 * x3 + 202,
                -400 * x2
            ],
            [
                0,
                -400 * x2,
                200
            ]
        ])