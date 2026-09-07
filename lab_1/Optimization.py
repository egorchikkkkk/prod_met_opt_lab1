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
        
    def optimize_CR(self, func, x0, a=1, tol=1e-3, max_iters=1000):
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
            
            x = x - a * grad
            
            history.append(x.copy())

            # формирование истории eps-радиуса для тестов
            if is_constructive:
                val_cr = func(x)
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

    def optimize_CR(self, func, x0, alpha=1, beta=0.5, gamma=2, tol=1e-3, max_iters=1000):
        is_constructive = isinstance(x0[0], ConstructiveNumber)

        if is_constructive:
            x0 = np.asarray(x0, dtype=object)
        else:
            x0 = np.asarray(x0, dtype=float)

        n = len(x0)

        def numeric_value(x):
            if isinstance(x, ConstructiveNumber):
                return x.get_real_value()
            return float(x)

        def point_to_numeric(point):
            if is_constructive:
                return np.array(
                    [numeric_value(x) for x in point],
                    dtype=float
                )

            return np.asarray(point, dtype=float)

        def function_numeric_value(value):
            return numeric_value(value)

        def make_constructive_point(point, eps):
            return np.array(
                [
                    ConstructiveNumber(x=float(value), eps=eps)
                    for value in point
                ],
                dtype=object
            )

        def point_eps(point):
            if not is_constructive:
                return 0.0

            return max(
                (x.b - x.a) / 2
                for x in point
            )

        history = [x0.copy()]
        eps_history = []

        simplex = [x0.copy()]

        for j in range(n):
            point = x0.copy()

            point[j] = point[j] + 1

            simplex.append(point)

        simplex = np.array(simplex, dtype=object if is_constructive else float)

        for iteration in range(max_iters):
            values = np.array(
                [func(point) for point in simplex],
                dtype=object if is_constructive else float
            )

            # Числовые значения нужны только для сортировки
            numeric_values = np.array(
                [function_numeric_value(value) for value in values],
                dtype=float
            )

            indices = np.argsort(numeric_values)

            simplex = simplex[indices]
            values = values[indices]
            numeric_values = numeric_values[indices]

            best = simplex[0]
            worst = simplex[-1]

            history.append(best.copy())

            if is_constructive:
                eps_history.append(point_eps(best))

            numeric_simplex = np.array(
                [point_to_numeric(point) for point in simplex]
            )

            # Максимальное расстояние от лучшей точки
            distances = np.linalg.norm(
                numeric_simplex - numeric_simplex[0],
                axis=1
            )

            simplex_size = np.max(distances)

            if simplex_size < tol:
                return OptimizationResult(
                    x=best,
                    value=values[0],
                    history=history,
                    iters=iteration + 1,
                    eps_history=eps_history
                )

            centroid = np.empty(n, dtype=object if is_constructive else float)

            for j in range(n):
                if is_constructive:
                    s = ConstructiveNumber(0, 0)

                    for k in range(n):
                        s = s + simplex[k][j]

                    centroid[j] = s / n

                else:
                    centroid[j] = np.mean(
                        simplex[:-1, j]
                    )

            reflected_dot = np.empty(
                n,
                dtype=object if is_constructive else float
            )

            for j in range(n):
                reflected_dot[j] = (
                    (1 + alpha) * centroid[j]
                    - alpha * worst[j]
                )

            reflected_val = func(reflected_dot)
            reflected_numeric = function_numeric_value(reflected_val)

            if reflected_numeric < numeric_values[0]:
                expanded_dot = np.empty(
                    n,
                    dtype=object if is_constructive else float
                )

                for j in range(n):
                    expanded_dot[j] = (
                        (1 - gamma) * centroid[j]
                        + gamma * reflected_dot[j]
                    )

                expanded_val = func(expanded_dot)
                expanded_numeric = function_numeric_value(expanded_val)

                if expanded_numeric < reflected_numeric:
                    simplex[-1] = expanded_dot
                else:
                    simplex[-1] = reflected_dot

            elif reflected_numeric < numeric_values[-2]:
                simplex[-1] = reflected_dot
                
            else:

                # Если отраженная точка хуже худшей,
                # используем худшую точку.
                if reflected_numeric >= numeric_values[-1]:
                    contraction_point = worst
                else:
                    contraction_point = reflected_dot

                contracted_dot = np.empty(
                    n,
                    dtype=object if is_constructive else float
                )

                for j in range(n):
                    contracted_dot[j] = (
                        beta * contraction_point[j]
                        + (1 - beta) * centroid[j]
                    )

                contracted_val = func(contracted_dot)
                contracted_numeric = function_numeric_value(contracted_val)

                if contracted_numeric < numeric_values[-1]:
                    simplex[-1] = contracted_dot

                else:
                    for k in range(1, n + 1):
                        new_point = np.empty(
                            n,
                            dtype=object if is_constructive else float
                        )

                        for j in range(n):
                            new_point[j] = (
                                simplex[0][j]
                                + (
                                    simplex[k][j]
                                    - simplex[0][j]
                                ) / 2
                            )

                        simplex[k] = new_point

        values = np.array(
            [func(point) for point in simplex],
            dtype=object if is_constructive else float
        )

        numeric_values = np.array(
            [function_numeric_value(value) for value in values],
            dtype=float
        )

        best_idx = np.argmin(numeric_values)

        return OptimizationResult(
            x=simplex[best_idx],
            value=values[best_idx],
            history=history,
            iters=max_iters,
            eps_history=eps_history
        )