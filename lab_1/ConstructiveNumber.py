from numbers import Real

class ConstructiveNumber:
    def __init__(self, a=None, b=None, x=None, eps=None):
        if isinstance(a, Real) and isinstance(b, Real):
            self.a = float(min(a, b))
            self.b = float(max(a, b))
        elif isinstance(x, Real) and isinstance(eps, Real):
            self.a = float(x - eps)
            self.b = float(x + eps)
        else:
            raise ValueError("The input must be (a, b) or (x, eps) numeric pair.")

    def __add__(self, other):
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a + other.a, self.b + other.b)
        elif isinstance(other, Real):
            return ConstructiveNumber(self.a + other, self.b + other)
        else:
            raise ValueError("Addition expects ConstructiveNumber or Real.")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a - other.b, self.b - other.a)
        elif isinstance(other, Real):
            return ConstructiveNumber(self.a - other, self.b - other)
        else:
            raise ValueError("Subtraction expects ConstructiveNumber or Real.")

    def __rsub__(self, other):
        return ConstructiveNumber(other - self.b, other - self.a)

    def __mul__(self, other):
        if isinstance(other, ConstructiveNumber):
            products = [
                self.a * other.a,
                self.a * other.b,
                self.b * other.a,
                self.b * other.b
            ]
    
            return ConstructiveNumber(
                min(products),
                max(products)
            )
    
        elif isinstance(other, Real):
            products = [
                self.a * other,
                self.b * other
            ]
    
            return ConstructiveNumber(
                min(products),
                max(products)
            )
        else:
            raise ValueError("Multiplication expects ConstructiveNumber or Real.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ConstructiveNumber):
            if other.a <= 0 <= other.b:
                raise ZeroDivisionError("Can not divide by number, which interval contains zero.")
                
            a = min(self.a / other.b, self.a / other.a, self.b / other.b, self.b / other.a)
            b = max(self.a / other.b, self.a / other.a, self.b / other.b, self.b / other.a)
            
            return ConstructiveNumber(a, b)
        elif isinstance(other, Real):
            if other == 0:
                raise ZeroDivisionError("Can not divide by zero.")
                
            a = min(self.a / other, self.b / other)
            b = max(self.a / other, self.b / other)
            
            return ConstructiveNumber(a, b)
        else:
            raise ValueError("Division expects ConstructiveNumber or Real.")

    def __rtruediv__(self, other):
        return ConstructiveNumber(other, other) / self

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Power must be a non-negative integer.")
        if power == 0:
            return ConstructiveNumber(1.0, 1.0)

        # для четных степеней не может получиться отрицательного значения
        if power % 2 == 0:
            candidates = [self.a**power, self.b**power]
            left = 0.0 if (self.a <= 0 <= self.b) else min(candidates)
            right = max(candidates)
            return ConstructiveNumber(left, right)
        # стандартная степень
        else:
            return ConstructiveNumber(self.a**power, self.b**power)

    def get_real_value(self, alpha=0.5):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha value should be between 0 and 1.")

        return (1 - alpha) * self.a + alpha * self.b

    def __eq__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a == other.a and self.b == other.b
        elif isinstance(other, Real):
            return self.a == other and self.b == other
        else:
            raise ValueError("Comparison expects ConstructiveNumber or Real.")

    def __lt__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.b < other.a
        elif isinstance(other, Real):
            return self.b < other
        else:
            raise ValueError("Comparison expects ConstructiveNumber or Real.")

    def __le__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.b <= other.a
        elif isinstance(other, Real):
            return self.b <= other
        else:
            raise ValueError("Comparison expects ConstructiveNumber or Real.")

    def __gt__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a > other.b
        elif isinstance(other, Real):
            return self.a > other
        else:
            raise ValueError("Comparison expects ConstructiveNumber or Real.")

    def __ge__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a >= other.b
        elif isinstance(other, Real):
            return self.a >= other
        else:
            raise ValueError("Comparison expects ConstructiveNumber or Real.")

    def __repr__(self):
        return f"CR({self.a}, {self.b})"