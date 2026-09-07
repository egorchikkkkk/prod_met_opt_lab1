from dataclasses import dataclass
from ConstructiveNumber import ConstructiveNumber
import numpy as np


@dataclass
class OptimizationResult:
    x: np.ndarray
    value: float | ConstructiveNumber
    iters: int
    history: list
    eps_history: list = None

class GradientDescent:
    def optimize(self, func, x0, a=1, tol=1e-3, max_iters=1000):
        is_constructive = isinstance(x0[0], ConstructiveNumber)
        dtype = object if is_constructive else float
        
        x = np.asarray(x0, dtype=dtype)

        history = [x.copy()]
        eps_history = []

        if is_constructive:
            input_eps = (x0[0].b - x0[0].a) / 2.0
            init_val_cr = func(x0)
            eps_history.append((init_val_cr.b - init_val_cr.a) / 2.0)
        else:
            input_eps = None

        for i in range(max_iters):
            grad = func.gradient(x)

            if is_constructive:
                numeric_grad = np.array([i.get_real_value() for i in grad])
            else:
                numeric_grad = grad
            
            if np.linalg.norm(numeric_grad) < tol:
                return OptimizationResult(
                    x=x,
                    value=func(x),
                    iters=i+1,
                    history=history,
                    eps_history=eps_history
                )
            
            x = x - a * numeric_grad
            
            history.append(x.copy())

            # формирование истории eps-радиуса для тестов
            if is_constructive:
                float_coords = [cr.get_real_value() for cr in x]
                cr_point = np.array([
                    ConstructiveNumber(
                        x=c, 
                        eps=input_eps
                    ) 
                    for c in float_coords
                ], dtype=object)
                val_cr = func(cr_point)
                eps_history.append((val_cr.b - val_cr.a) / 2.0)

        return OptimizationResult(
                    x=x,
                    value=func(x),
                    iters=max_iters,
                    history=history,
                    eps_history=eps_history
                )

class DownhillSimplexMethod:
    def optimize(self, func, x0, alpha=1, beta=0.5, gamma=2, tol=1e-3, max_iters=1000):
        is_constructive = isinstance(x0[0], ConstructiveNumber)
        
        if is_constructive:
            # преобразуем к float когда получаем конструктивное
            numeric_x0 = np.array([num.get_real_value() for num in x0], dtype=float)
            # радиус интервала каждой координаты
            init_eps = np.array([(num.b - num.a) / 2 for num in x0], dtype=float)
            # фиксируем первый радиус для инициализации истории
            input_eps = init_eps[0]
        else:
            numeric_x0 = np.asarray(x0, dtype=float)
            init_eps = None
            input_eps = None

        n = len(x0)

        history = [numeric_x0.copy()]
        eps_history = []

        # строим симплекс
        simplex = [numeric_x0.copy()]
        for i in range(n):
            dot = numeric_x0.copy()
            dot[i] += 1
            simplex.append(dot)

        simplex = np.array(simplex, dtype=float)

        for i in range(max_iters):

            # Шаг 1
            # вычисляем значения ф-ии в выбранных точках
            values = np.array([func(dot) for dot in simplex], dtype=float)

            # сортируем
            indices = np.argsort(values)

            simplex = simplex[indices]
            values = values[indices]

            best = simplex[0]
            worst = simplex[-1]

            history.append(best.copy())

            # формирование истории eps-радиуса для тестов
            if is_constructive:
                cr_best = np.array([
                    ConstructiveNumber(
                        x=val, 
                        eps=input_eps
                    )
                    for val in best
                ], dtype=object)
                val_cr = func(cr_best)
                eps_history.append((val_cr.b - val_cr.a) / 2.0)

            # Критерий сходимости:
            # максимальное расстояние до лучшей точки
            simplex_size = np.max(
                np.linalg.norm(simplex - best, axis=1)
            )

            if simplex_size < tol:

                if is_constructive:
                    result_x = np.array([
                        ConstructiveNumber(
                            x=value,
                            eps=init_eps[i]
                        )
                        for i, value in enumerate(best)
                    ], dtype=object)
                else:
                    result_x = best

                return OptimizationResult(
                    x=result_x,
                    value=func(best),
                    iters=i+1,
                    history=history,
                    eps_history=eps_history
                )

            # Шаг 2
            # считаем центр масс симплекса
            centroid = np.mean(simplex[:-1], axis=0)

            # Шаг 3
            # отражаем худшую точку max_dot относительно centroid
            reflected_dot = (1 + alpha) * centroid - alpha * worst
            reflected_val = func(reflected_dot)

            # проверяем лучше ли отраженная точка, чем лучшая до этого
            if reflected_val < values[0]:
                # растягиваем
                expanded_dot = (1 - gamma) * centroid + gamma * reflected_dot
                expanded_val = func(expanded_dot)

                # если значение в растянутой точке меньше, чем в отраженной, то можно растянуть сиплекс до нее
                # меняем худшую точку симплекса на расширенную
                if expanded_val < reflected_val:
                    simplex[-1] = expanded_dot

                # если значение в отраженной точке меньше, чем в растянутой, то переместились слишком далеко
                # меняем худшую точку симплекса на отраженную
                elif reflected_val < expanded_val:
                    simplex[-1] = reflected_dot

            # проверяем лучше ли отраженная точка, чем 2-ая худшая до этого
            # если да, то это неплохая точка, меняем худшую на нее
            elif reflected_val < values[-2]:
                simplex[-1] = reflected_dot

            else:
                # проверяем лежит ли значение в отраженной точке между предыдущими 2-мя худшими точками
                # если да, то меняем месатми отраженную и худшую
                if values[-2] < reflected_val < values[-1]:
                    reflected_dot, simplex[-1] = simplex[-1], reflected_dot
                    reflected_val, values[-1] = values[-1], reflected_val

                # вычисляем сжатую точку
                contracted_dot = beta * simplex[-1] + (1 - beta) * centroid
                contracted_val = func(contracted_dot)

                if contracted_val < values[-1]:
                    simplex[-1] = contracted_dot

                # если значение в сжатой точке больще, чем в худшей то первоначальные точки оказались самыми удачными
                # делаем глобальное сжатие симплекса
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + (simplex[i] - simplex[0]) / 2

        values = np.array([func(dot) for dot in simplex], dtype=float)
        indices = np.argsort(values)
        best = simplex[indices[0]]

        if is_constructive:
            result_x = np.array([
                ConstructiveNumber(
                    x=value,
                    eps=init_eps[i]
                )
                for i, value in enumerate(best)
            ], dtype=object)
        else:
            result_x = best

        return OptimizationResult(
            x=result_x,
            value=values[indices[0]],
            iters=max_iters,
            history=history,
            eps_history=eps_history
        )