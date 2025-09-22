"""
Crop Rotation Optimization System
Generated with assistance from DeepSeek AI

Author: WorkFreeBoyar
AI Assistant: DeepSeek
Date: 22 september 2025
"""
"""
This algorithm is prior art published on 22 september 2025 and cannot be patented.
"""
from collections import namedtuple
import numpy as np
import math
from functools import reduce

#узнаем желанную операцию
while True:
    try:
        UserInput = int(input("1 - создать файл с общими ошибками; 2 - на основании файла посчитать годовые ошибки; 12 - сначала 1, потом 2 : "))
        if UserInput in (1, 2, 12):
            break
        else:
            print("некорректный input")
    except ValueError:
        print("некорректный input (введите число)")

# 1. СЛОВАРЬ КУЛЬТУР
# Создаем словарь, где ключ - числовой идентификатор культуры, значение - ее название.
# Это позволяет легко добавлять и изменять культуры.
Crops = {
    1: "огурец",
    2: "помидор",
    3: "перцы",
    4: "укроп или чеснок или горчица",
    5: "фасоль",
    6: "капуста",
    7: "хрен",
    8: "табак или кориандр(кинза)"
    # TODO: Добавьте сюда другие культуры по образцу
    # <числовой_id>: "<Название культуры>",
}
# Для удобства создадим обратный словарь (название -> id), он может пригодиться
CropsReverse = {v: k for k, v in Crops.items()}

# 2. МАТРИЦА СЕВООБОРОТОВ Mcrops
# Создаем массив массивов (список списков).
# Каждый внутренний список - это последовательность культур (их числовых id) в севообороте.
Mcrops = [
    [1,3,5],
    [1,2,4],
    [1,3,5],
    [1,2,5],
    [1,3,7],
    [1,2,7],
    [1,8,7],
    [1,8,5],
    [1,8,4],
    [1,5,4],
    [1,5,6],
    [1,6,4],
    [1,3,6],
    [1,2,6],
    # TODO: Добавьте сюда другие схемы севооборота
]
# 6. ПАРАМЕТРЫ АЛГОРИТМА
N = 20       # Количество используемых линий севооборота в матрице
Steps = 50 # Количество шагов алгоритма Multi-Start
MAX_ERROR = 0.07  # Максимальная допустимая общая ошибка для дальнейшей оптимизации по годам  

# 4. МАССИВ ОПТИМАЛЬНЫХ ПРОПОРЦИЙ OptimalCropQuantity
# Создаем массив оптимальных долей для каждой культуры.
# Размер массива равен количеству культур в словаре Crops.
# Индекс в массиве соответствует числовому id культуры.
# Важно: Сумма всех значений в этом массиве должна быть равна 1.0.
OptimalCropQuantity = np.zeros(max(Crops.keys()) + 1, dtype=float)  # Создаем массив с запасом

# Заполняем массив оптимальных пропорций.
# Пример: хотим, чтобы 30% урожая был огурец, 25% - помидор и т.д.
OptimalCropQuantity[1] = 1/3  # Огурец
OptimalCropQuantity[2] = 0.22  # Помидор
OptimalCropQuantity[3] = 1/6  # перцы
OptimalCropQuantity[4] = 0.10  # укроп иили чеснок или горчица
OptimalCropQuantity[5] = 0.07  # фасоль
OptimalCropQuantity[6] = 0.07  # капуста
OptimalCropQuantity[7] = 0.04  # хрен
OptimalCropQuantity[8] = 0  # Табак
# TODO: Отрегулируйте значения так, чтобы их сумма была равна 1.0
# Проверка суммы (опционально, но рекомендуется)
assert abs(np.sum(OptimalCropQuantity) - 1.0) < 1e-10, "Сумма OptimalCropQuantity должна быть равна 1.0"

CropErrorWeights = np.ones(max(Crops.keys()) + 1, dtype=float)

# Задаем веса вручную. Пример: ошибка по огурцу и помидору очень критична,
# по капусте - менее критична, по луку - почти не важна.
CropErrorWeights[1] = 5.0  # Огурец (высший приоритет)
CropErrorWeights[2] = 3.5  # Помидор
CropErrorWeights[3] = 3.0  # перцы
CropErrorWeights[4] = 6.0  # укроп или чеснок или горчица
CropErrorWeights[5] = 1.5  # фасоль
CropErrorWeights[6] = 1.5  # капуста
CropErrorWeights[7] = 1  # хрен
CropErrorWeights[8] = 0.5  # Табак
# 5. ОБЪЯВЛЯЕМ СТРУКТУРУ ДЛЯ РЕШЕНИЯ
CropRotationLine = namedtuple('CropRotationLine', ['m', 'fi'])

# 7. ИНИЦИАЛИЗИРУЕМ ГЛОБАЛЬНЫЕ СТРУКТУРЫ ДЛЯ ХРАНЕНИЯ ИСТОРИИ
L_hist = []  # Будет хранить найденные оптимизированные решения L
H_hist = []  # Будет хранить гистограммы этих решений

# 3. МАССИВ ДЛИН СЕВООБОРОТОВ T
# Автоматически создаем массив T на основе данных Mcrops.
# T[i] равна длине (количеству культур) в севообороте Mcrops[i]
T = np.array([len(rotation) for rotation in Mcrops])
# Теперь M тоже можно было бы определить как len(Mcrops), но оставим как заданную константу для ясности
M = len(Mcrops)

def lcm(a, b):
    """Вычисляет НОК двух чисел."""
    return abs(a * b) // math.gcd(a, b) if a and b else 0

def lcm_of_list(numbers):
    """Вычисляет НОК для списка чисел."""
    return reduce(lcm, numbers)

# Вычисляем N2 - общий период, кратный всем периодам севооборотов
N2 = lcm_of_list(T)
print(f"Рассчитанный N2 (НОК периодов {T}): {N2}")

def calculate_current_crop_quantity(l_solution):
    """
    Вычисляет относительное количество каждой культуры в данном комплексе севооборотов L.
    Возвращает массив того же размера, что и OptimalCropQuantity, с нормализованными долями.

    Args:
        l_solution: Список из N элементов типа CropRotationLine (решение L).

    Returns:
        np.ndarray: Массив с рассчитанными долями культур.
    """

    # Создаем буферный массив для суммирования вклада каждой культуры.
    # Размер: max_id + 1, чтобы индексы совпадали с Crops.keys()
    total_occurrences = np.zeros(max(Crops.keys()) + 1, dtype=float)

    # Проходим по каждой линии севооборота в решении L
    for line in l_solution:
        m_index = line.m
        phase = line.fi
        rotation_length = T[m_index]  # Длина текущего севооборота
        rotation_sequence = Mcrops[m_index]  # Последовательность культур в севообороте

        # Коэффициент повторения цикла за общий период N2
        cycles_in_period = N2 / rotation_length
        # Вклад ОДНОЙ культуры этого севооборота за ОДИН цикл равен 1.
        # Поэтому общий вклад всей строки равен cycles_in_period.
        # Но нам нужно распределить этот вклад по конкретным культурам.

        # Создаем временный массив для вклада культур из ЭТОГО севооборота
        crop_contribution = np.zeros_like(total_occurrences, dtype=float)

        # Проходим по всем позициям в севообороте и учитываем культуру на этой позиции
        for year_in_rotation in range(rotation_length):
            # Вычисляем "глобальную" позицию с учетом фазы
            # (это нужно только для правильного учета количества каждой культуры)
            effective_index = (year_in_rotation + phase) % rotation_length
            crop_id = rotation_sequence[effective_index]
            # Каждая культура в севообороте вносит вклад 1 за один полный цикл
            # За N2 лет цикл повторится cycles_in_period раз.
            # Поэтому общий вклад этой конкретной культуры равен cycles_in_period.
            crop_contribution[crop_id] += 1.0 * cycles_in_period
        # Добавляем вклад от текущего севооборота к общему вкладу
        total_occurrences += crop_contribution

    # Нормализуем массив, чтобы сумма всех долей была равна 1
    total_sum = np.sum(total_occurrences)
    if total_sum > 0:
        current_quantity = total_occurrences / total_sum
    else:
        current_quantity = total_occurrences  # Вернет нулевой массив

    return current_quantity

# 8. ФУНКЦИЯ ГЕНЕРАЦИИ СЛУЧАЙНОГО РЕШЕНИЯ (ОБНОВЛЕННАЯ)
def generate_random_l():
    """
    Генерирует случайное начальное решение L.
    L - это вектор длины N, каждый элемент которого является парой (m, fi).
    m выбирается случайно из [0, M-1], fi выбирается случайно из [0, T[m]-1].
    """
    new_l = []
    for _ in range(N):
        random_m = np.random.randint(0, M)          # Выбираем случайный севооборот
        random_fi = np.random.randint(0, T[random_m]) # Выбираем случайную фазу для него
        new_l.append(CropRotationLine(m=random_m, fi=random_fi))
    return new_l

def generate_l():
    """
    Генерирует новое начальное решение L, которое максимально отличается
    от всех уже исследованных решений в H_hist.
    """
    # Если история пустая, генерируем полностью случайное решение
    if not H_hist:
        return generate_random_l()
    
    # Генерируем пул кандидатных гистограмм
    num_candidates = 30  # Количество случайных кандидатов для выбора
    candidates = []
    
    for _ in range(num_candidates):
        # Генерируем случайную гистограмму, где сумма элементов равна N
        # Это эквивалентно случайному распределению N шаров по M урнам
        hist_candidate = np.zeros(M, dtype=int)
        
        # Распределяем N "единиц" по M позициям
        for _ in range(N):
            random_bin = np.random.randint(0, M)
            hist_candidate[random_bin] += 1
        
        candidates.append(hist_candidate)
    
    # Находим кандидата с максимальным минимальным расстоянием до всех известных гистограмм
    best_candidate = None
    best_min_distance = -1
    
    for candidate in candidates:
        # Вычисляем минимальное расстояние от этого кандидата до всех известных гистограмм
        min_distance = float('inf')
        for known_hist in H_hist:
            distance = np.sum(np.abs(candidate - known_hist))  # Манхэттенское расстояние
            if distance < min_distance:
                min_distance = distance
        
        # Выбираем кандидата с наибольшим минимальным расстоянием
        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_candidate = candidate
    
    # Преобразуем лучшую гистограмму в решение L
    new_l = []
    for m_index in range(M):
        count = best_candidate[m_index]
        # Добавляем count раз севооборот с индексом m_index
        for _ in range(count):
            # Для каждого добавления выбираем случайную фазу
            random_fi = np.random.randint(0, T[m_index])
            new_l.append(CropRotationLine(m=m_index, fi=random_fi))
    
    # Перемешиваем решение, чтобы порядок строк не влиял на дальнейшую оптимизацию
    np.random.shuffle(new_l)
    
    return new_l

# 9. ФУНКЦИЯ ВЫЧИСЛЕНИЯ ГИСТОГРАММЫ (БЕЗ ИЗМЕНЕНИЙ)
def calculate_histogram(l_solution):
    """
    Вычисляет гистограмму для решения L.
    Гистограмма - это вектор длины M, где каждый элемент H[i]
    показывает, сколько раз севооборот с индексом i встречается в решении L.
    """
    hist = np.zeros(M, dtype=int)
    for line in l_solution:
        m_index = line.m
        hist[m_index] += 1
    return hist

def calculate_histogram_errors():
    """
    Вычисляет ошибки для всех гистограмм в H_hist и сохраняет в отдельный массив.
    Для вычисления ошибки создается временное решение L на основе гистограммы.
    
    Returns:
        list: Массив ошибок для каждой гистограммы в H_hist.
    """
    histogram_errors = []
    
    print("Вычисление ошибок для всех найденных гистограмм...")
    
    for i, hist in enumerate(H_hist):
        # Создаем временное решение L на основе гистограммы
        temp_solution = []
        for m_index in range(M):
            count = hist[m_index]
            # Добавляем count раз севооборот с индексом m_index со случайной фазой
            for _ in range(count):
                random_fi = np.random.randint(0, T[m_index])
                temp_solution.append(CropRotationLine(m=m_index, fi=random_fi))
        
        # Вычисляем ошибку для этого решения
        error = calculate_error(temp_solution)
        histogram_errors.append(error)
        
        if (i + 1) % 10 == 0 or i == 0 or i == len(H_hist) - 1:
            print(f"  Гистограмма {i + 1}/{len(H_hist)}: ошибка = {error:.6f}")
    
    return histogram_errors

# 10. ЗАГОТОВКА ДЛЯ ФУНКЦИИ ОШИБКИ (ЗАГЛУШКА)
def calculate_error(l_solution):
    """
    Вычисляет общую взвешенную ошибку для решения L.
    Ошибка рассчитывается как сумма взвешенных квадратов отклонений
    от целевого распределения по всем культурам.

    Args:
        l_solution: Список из N элементов типа CropRotationLine (решение L).

    Returns:
        float: Общая взвешенная ошибка решения.
    """

    # Шаг 1: Рассчитываем текущее распределение культур для решения L
    current_quantity = calculate_current_crop_quantity(l_solution)

    # Шаг 2: Вычисляем отклонение для каждой культуры
    # Берем квадрат отклонения (более строгая "штрафная" функция)
    deviation = current_quantity - OptimalCropQuantity
    squared_error = deviation ** 2

    # Шаг 3: Применяем весовые коэффициенты
    weighted_error = squared_error * CropErrorWeights

    # Шаг 4: Суммируем ошибку по всем культурам
    total_error = np.sum(weighted_error)

    return total_error

def calculate_yearly_error(l_solution):
    """
    Вычисляет ошибку по годам для решения L.
    Возвращает вектор ошибок длины N2 и суммарную ошибку по всем годам.
    
    Args:
        l_solution: Решение L (список из CropRotationLine).
    
    Returns:
        tuple: (yearly_errors, total_yearly_error) где:
            - yearly_errors: np.ndarray ошибок для каждого года (длина N2)
            - total_yearly_error: float - сумма ошибок по всем годам
    """
    # Создаем матрицу культур по годам: N2 лет × количество культур
    yearly_crop_quantity = np.zeros((N2, len(Crops) + 1), dtype=float)  # +1 для индексации с 1
    
    # Заполняем матрицу: для каждого года считаем количество каждой культуры
    for line in l_solution:
        m_index = line.m
        phase = line.fi
        rotation = Mcrops[m_index]
        rotation_length = len(rotation)
        
        # Для каждого года в периоде N2 определяем, какая культура высаживается
        for year in range(N2):
            # Вычисляем позицию в севообороте для данного года
            pos_in_rotation = (year + phase) % rotation_length
            crop_id = rotation[pos_in_rotation]
            
            # Увеличиваем счетчик этой культуры для данного года
            yearly_crop_quantity[year, crop_id] += 1
    
    # Нормализуем: преобразуем количества в доли для каждого года
    for year in range(N2):
        total_crops_in_year = np.sum(yearly_crop_quantity[year, :])
        if total_crops_in_year > 0:
            yearly_crop_quantity[year, :] /= total_crops_in_year
    
    # Вычисляем ошибку для каждого года
    yearly_errors = np.zeros(N2, dtype=float)
    
    for year in range(N2):
        # Получаем распределение культур для этого года
        current_distribution = yearly_crop_quantity[year, 1:]  # пропускаем индекс 0
        
        # Вычисляем отклонение от целевого распределения
        deviation = current_distribution - OptimalCropQuantity[1:]
        
        # Вычисляем взвешенную квадратичную ошибку для этого года
        squared_error = deviation ** 2
        weighted_error = squared_error * CropErrorWeights[1:]  # применяем веса
        yearly_errors[year] = np.sum(weighted_error)
    
    # Суммируем ошибки по всем годам
    total_yearly_error = np.sum(yearly_errors)
    
    return yearly_errors, total_yearly_error

def calculate_total_yearly_error(l_solution):
    """
    Вспомогательная функция: возвращает только суммарную ошибку по годам.
    """
    _, total_error = calculate_yearly_error(l_solution)
    return total_error

# 11. ЖАДНЫЙ СПУСК 
def greedy_descent(l_input):
    """
    Алгоритм жадного покоординатного спуска для оптимизации состава севооборотов.
    Сохраняет в L_hist решение после каждой полной итерации улучшения.

    Args:
        l_input: Исходное решение L (список из CropRotationLine).

    Returns:
        Оптимизированное решение L_optimized.
    """
    # Создаем копию входного решения для модификации
    current_solution = l_input.copy()
    current_error = calculate_error(current_solution)
    
    # Сохраняем начальное решение в историю
    L_hist.append(current_solution.copy())
    H_hist.append(calculate_histogram(current_solution))
    print(f"Начальная ошибка: {current_error:.6f}")
    
    improved = True
    iteration = 0
    
    # Главный цикл: продолжаем, пока находится улучшение
    while improved:
        improved = False
        iteration += 1
        
        # Проходим по каждому элементу в решении L
        for i in range(len(current_solution)):
            best_m_for_i = current_solution[i].m
            best_error_for_i = current_error
            
            # Сохраняем исходное состояние элемента i
            original_line = current_solution[i]
            
            # Перебираем все возможные севообороты m для этой позиции
            for candidate_m in range(M):
                # Пропускаем текущий m
                if candidate_m == original_line.m:
                    continue
                    
                # Создаем пробное решение с candidate_m на позиции i
                trial_line = CropRotationLine(m=candidate_m, fi=original_line.fi)
                current_solution[i] = trial_line
                
                # Вычисляем ошибку для пробного решения
                trial_error = calculate_error(current_solution)
                
                # Если нашли лучшее значение для этой позиции
                if trial_error < best_error_for_i:
                    best_error_for_i = trial_error
                    best_m_for_i = candidate_m
                    improved = True  # Запомним, что было улучшение в этой итерации
            
            # Возвращаем лучший найденный вариант для позиции i
            best_line = CropRotationLine(m=best_m_for_i, fi=original_line.fi)
            current_solution[i] = best_line
            current_error = best_error_for_i
        
        # После полного прохода по всем элементам сохраняем текущее состояние
        # (только если было какое-то улучшение в этой итерации)
        if improved:
            L_hist.append(current_solution.copy())
            H_hist.append(calculate_histogram(current_solution))
            print(f"  Итерация {iteration}, ошибка: {current_error:.6f}")
    
    print(f"Локальный минимум достигнут. Финальная ошибка: {current_error:.6f}")
    return current_solution

def optimize_yearly_error(l_input):
    """
    Оптимизирует решение L для минимизации ошибки по годам.
    Для каждого элемента L[i] перебирает все комбинации (m, fi)
    и выбирает ту, которая максимально уменьшает суммарную ошибку по годам.
    """
    # Создаем копию входного решения для модификации
    current_solution = l_input.copy()
    current_error = calculate_total_yearly_error(current_solution)
    
    improved = True
    iteration = 0
    
    print(f"  Начальная ошибка по годам: {current_error:.6f}")
    
    # Главный цикл: продолжаем, пока находится улучшение
    while improved:
        improved = False
        iteration += 1
        
        # Проходим по каждому элементу в решении L
        for i in range(len(current_solution)):
            best_m_for_i = current_solution[i].m
            best_fi_for_i = current_solution[i].fi
            best_error_for_i = current_error
            
            # Сохраняем исходное состояние элемента i
            original_line = current_solution[i]
            
            # Перебираем все возможные комбинации (m, fi) для этой позиции
            for candidate_m in range(M):
                for candidate_fi in range(T[candidate_m]):
                    # Пропускаем текущую комбинацию
                    if candidate_m == original_line.m and candidate_fi == original_line.fi:
                        continue
                    
                    # Создаем пробное решение с candidate_m и candidate_fi на позиции i
                    trial_line = CropRotationLine(m=candidate_m, fi=candidate_fi)
                    current_solution[i] = trial_line
                    
                    # Вычисляем ошибку по годам для пробного решения
                    trial_error = calculate_total_yearly_error(current_solution)
                    
                    # Если нашли лучшее значение для этой позиции
                    if trial_error < best_error_for_i:
                        best_error_for_i = trial_error
                        best_m_for_i = candidate_m
                        best_fi_for_i = candidate_fi
                        improved = True
            
            # Возвращаем лучший найденный вариант для позиции i
            best_line = CropRotationLine(m=best_m_for_i, fi=best_fi_for_i)
            current_solution[i] = best_line
            
            # Обновляем текущую ошибку для следующих позиций
            current_error = best_error_for_i
        
        # После полного прохода сохраняем текущее состояние
        if improved:
            L_hist_yearly.append(current_solution.copy())
            H_hist_yearly.append(calculate_histogram(current_solution))
            yearly_errors_hist.append(current_error)
            print(f"  Итерация {iteration}, ошибка по годам: {current_error:.6f}")
    
    print(f"  Локальный минимум по годам достигнут. Финальная ошибка: {current_error:.6f}")
    return current_solution
def remove_duplicate_yearly_solutions():
    """
    Удаляет дубликаты решений из L_hist_yearly и H_hist_yearly.
    """
    global L_hist_yearly, H_hist_yearly, yearly_errors_hist
    
    print("Поиск и удаление дубликатов решений (оптимизация по годам)...")
    print(f"До очистки: {len(L_hist_yearly)} решений")
    
    unique_solutions = {}
    indices_to_keep = []
    
    for i, hist in enumerate(H_hist_yearly):
        hist_tuple = tuple(hist)
        
        if hist_tuple not in unique_solutions:
            unique_solutions[hist_tuple] = i
            indices_to_keep.append(i)
    
    L_hist_yearly = [L_hist_yearly[i] for i in indices_to_keep]
    H_hist_yearly = [H_hist_yearly[i] for i in indices_to_keep]
    yearly_errors_hist = [yearly_errors_hist[i] for i in indices_to_keep]
    
    print(f"После очистки: {len(L_hist_yearly)} уникальных решений")
    return len(indices_to_keep)

def save_yearly_results_to_file(filename="yearly_optimization_results.txt"):
    """
    Сохраняет результаты оптимизации по годам в файл.
    Сохраняет полные решения L в формате: m:fi m:fi m:fi ... ; error
    """
    # Удаляем дубликаты
    unique_count = remove_duplicate_yearly_solutions()
    
    print(f"Сохранение {unique_count} уникальных результатов в файл {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# M={M}, N={N}, Steps={Steps}, N2={N2}\n")
        f.write(f"# Crop IDs: {Crops}\n")
        f.write(f"# Mcrops: {Mcrops}\n")
        f.write(f"# T: {T}\n")
        f.write(f"# Unique solutions (yearly optimization): {unique_count}\n")
        f.write(f"# Format: m1:fi1 m2:fi2 m3:fi3 ... ; yearly_error\n")
        f.write("#" + "="*50 + "\n")
        
        for solution, error in zip(L_hist_yearly, yearly_errors_hist):
            # Формируем строку с решением
            solution_str = " ".join(f"{line.m}:{line.fi}" for line in solution)
            f.write(f"{solution_str} ; {error:.10f}\n")
    
    print(f"Результаты оптимизации по годам сохранены в {filename}")

def load_yearly_results_from_file(filename="yearly_optimization_results.txt"):
    """
    Загружает решения и ошибки из файла оптимизации по годам.
    """
    loaded_solutions = []
    loaded_errors = []
    
    print(f"Загрузка результатов из файла {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Пропускаем комментарии и пустые строки
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                # Разделяем строку на решение и ошибку
                if ';' in line:
                    solution_part, error_part = line.split(';', 1)
                    
                    # Парсим решение
                    solution_lines = []
                    for item in solution_part.strip().split():
                        if ':' in item:
                            m_str, fi_str = item.split(':', 1)
                            try:
                                m_val = int(m_str)
                                fi_val = int(fi_str)
                                # Проверяем, что фаза допустима для данного севооборота
                                if 0 <= fi_val < T[m_val]:
                                    solution_lines.append(CropRotationLine(m=m_val, fi=fi_val))
                                else:
                                    print(f"Предупреждение: Недопустимая фаза {fi_val} для севооборота {m_val}")
                            except ValueError:
                                print(f"Предупреждение: Не удалось распарсить '{item}'")
                    
                    # Проверяем, что решение имеет правильную длину
                    if len(solution_lines) == N:
                        loaded_solutions.append(solution_lines)
                        loaded_errors.append(float(error_part.strip()))
                    else:
                        print(f"Предупреждение: Пропущена строка с неправильным количеством элементов: {len(solution_lines)} вместо {N}")
        
        print(f"Загружено {len(loaded_solutions)} решений")
        
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
    
    return loaded_solutions, loaded_errors

def remove_duplicate_solutions():
    """
    Удаляет дубликаты решений из L_hist и H_hist на основе сравнения гистограмм.
    Сохраняет только уникальные гистограммы и соответствующие им решения.
    """
    global L_hist, H_hist
    
    print("Поиск и удаление дубликатов решений...")
    print(f"До очистки: {len(L_hist)} решений в L_hist, {len(H_hist)} гистограмм в H_hist")
    
    # Создаем словарь для отслеживания уникальных гистограмм
    unique_solutions = {}
    indices_to_keep = []
    
    # Проходим по всем решениям и находим уникальные гистограммы
    for i, hist in enumerate(H_hist):
        # Преобразуем гистограмму в кортеж для использования как ключ словаря
        hist_tuple = tuple(hist)
        
        if hist_tuple not in unique_solutions:
            # Сохраняем индекс первого вхождения этой гистограммы
            unique_solutions[hist_tuple] = i
            indices_to_keep.append(i)
    
    # Создаем новые списки только с уникальными решениями
    L_hist_unique = [L_hist[i] for i in indices_to_keep]
    H_hist_unique = [H_hist[i] for i in indices_to_keep]
    
    # Заменяем исходные списки
    L_hist = L_hist_unique
    H_hist = H_hist_unique
    
    print(f"После очистки: {len(L_hist)} уникальных решений в L_hist, {len(H_hist)} уникальных гистограмм в H_hist")
    
    return len(indices_to_keep)

def save_results_to_file(filename="optimization_results.txt"):
    """
    Сохраняет гистограммы и соответствующие ошибки в текстовый файл.
    Перед сохранением удаляет дубликаты решений.
    """
    # Удаляем дубликаты
    unique_count = remove_duplicate_solutions()
    
    # Вычисляем ошибки для всех уникальных гистограмм
    histogram_errors = calculate_histogram_errors()
    
    print(f"Сохранение {unique_count} уникальных результатов в файл {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Записываем заголовок с метаинформацией
        f.write(f"# M={M}, N={N}, Steps={Steps}, N2={N2}\n")
        f.write(f"# Crop IDs: {Crops}\n")
        f.write(f"# Unique solutions: {unique_count}\n")
        f.write(f"# Format: histogram (space-separated) ; error\n")
        f.write("#" + "="*50 + "\n")
        
        # Записываем данные
        for hist, error in zip(H_hist, histogram_errors):
            # Преобразуем гистограмму в строку с разделителями-пробелами
            hist_str = " ".join(str(x) for x in hist)
            # Записываем в формате: "1 2 3 4 5 ; 0.123456"
            f.write(f"{hist_str} ; {error:.10f}\n")
    
    print(f"Уникальные результаты сохранены в {filename}")

# Также обновим функцию load_results_from_file для более надежной работы
def load_results_from_file(filename="optimization_results.txt"):
    """
    Загружает гистограммы и ошибки из файла.
    """
    loaded_H_hist = []
    loaded_errors = []
    
    print(f"Загрузка результатов из файла {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Пропускаем комментарии и пустые строки
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                # Разделяем строку на гистограмму и ошибку
                if ';' in line:
                    hist_part, error_part = line.split(';', 1)
                    
                    # Парсим гистограмму
                    hist_values = [int(x) for x in hist_part.split()]
                    loaded_H_hist.append(np.array(hist_values, dtype=int))
                    
                    # Парсим ошибку
                    loaded_errors.append(float(error_part.strip()))
        
        print(f"Загружено {len(loaded_H_hist)} уникальных гистограмм")
        
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Будет создан новый.")
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
    
    return loaded_H_hist, loaded_errors

if (UserInput==12 or UserInput==1):
    #12. ОСНОВНОЙ ЦИКЛ АЛГОРИТМА (БЕЗ ИЗМЕНЕНИЙ)
    print(f"Запуск Multi-Start алгоритма на {Steps} шагов...")

    for step in range(Steps):
        current_l = generate_l()
        optimized_l = greedy_descent(current_l)
        optimized_hist = calculate_histogram(optimized_l)

        L_hist.append(optimized_l)
        H_hist.append(optimized_hist)

    print(optimized_l,'\n',optimized_hist,'\n')
    print("Алгоритм завершил работу.")
    histogram_errors=calculate_histogram_errors()
    remove_duplicate_solutions()
    save_results_to_file("matrix_optimization_results.txt")

def final_optimization_stage(max_error=0.1, input_filename="matrix_optimization_results.txt", 
                           output_filename="final_optimization_results.txt"):
    """
    Заключительный этап оптимизации:
    1. Загружает гистограммы с ошибкой <= max_error
    2. Создает решения L_exam со случайными фазами
    3. Проводит оптимизацию по годам для каждого решения
    4. Сохраняет результаты и выводит лучшее решение
    """
    print("=" * 60)
    print("ЗАКЛЮЧИТЕЛЬНЫЙ ЭТАП ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    # 1. Загружаем результаты из файла
    print("1. Загрузка результатов предыдущего этапа...")
    loaded_H_hist, loaded_errors = load_results_from_file(input_filename)
    
    if not loaded_H_hist:
        print("Ошибка: не удалось загрузить результаты из файла")
        return
    
    print(f"Загружено {len(loaded_H_hist)} гистограмм")
    
    # 2. Отбираем гистограммы с ошибкой <= max_error
    print(f"2. Отбор гистограмм с ошибкой <= {max_error}...")
    H_exam = []
    errors_exam = []
    
    for hist, error in zip(loaded_H_hist, loaded_errors):
        if error <= max_error:
            H_exam.append(hist)
            errors_exam.append(error)
    
    print(f"Отобрано {len(H_exam)} гистограмм для дальнейшей оптимизации")
    
    if not H_exam:
        print("Нет гистограмм, удовлетворяющих критерию отбора")
        return
    
    # 3. Создаем L_exam из отобранных гистограмм со случайными фазами
    print("3. Создание решений L_exam со случайными фазами...")
    L_exam = []
    
    for i, hist in enumerate(H_exam):
        solution = []
        for m_index in range(M):
            count = hist[m_index]
            # Добавляем count раз севооборот с индексом m_index со случайной фазой
            for _ in range(count):
                random_fi = np.random.randint(0, T[m_index])
                solution.append(CropRotationLine(m=m_index, fi=random_fi))
        
        # Перемешиваем решение
        np.random.shuffle(solution)
        L_exam.append(solution)
        
        if (i + 1) % 10 == 0 or i == 0 or i == len(H_exam) - 1:
            print(f"  Создано решение {i + 1}/{len(H_exam)}")
    
    # 4. Проводим оптимизацию по годам для каждого L_exam
    print("4. Оптимизация по годам для отобранных решений...")
    
    # Очищаем глобальные переменные для нового этапа
    global L_hist_yearly, H_hist_yearly, yearly_errors_hist
    L_hist_yearly = []
    H_hist_yearly = []
    yearly_errors_hist = []
    
    for i, l_solution in enumerate(L_exam):
        print(f"  Оптимизация решения {i + 1}/{len(L_exam)}...")
        
        # Вычисляем начальную ошибку
        initial_error = calculate_total_yearly_error(l_solution)
        print(f"    Начальная ошибка по годам: {initial_error:.6f}")
        
        # Проводим оптимизацию
        optimized_solution = optimize_yearly_error(l_solution)
        
        # Сохраняем результат
        optimized_hist = calculate_histogram(optimized_solution)
        final_error = calculate_total_yearly_error(optimized_solution)
        
        L_hist_yearly.append(optimized_solution)
        H_hist_yearly.append(optimized_hist)
        yearly_errors_hist.append(final_error)
        
        print(f"    Финальная ошибка по годам: {final_error:.6f}")
        print(f"    Улучшение: {initial_error - final_error:.6f}")
    
    # 5. Сохраняем результаты заключительного этапа
    print("5. Сохранение результатов...")
    save_yearly_results_to_file(output_filename)
    
    # 6. Находим и выводим лучшее решение
    print("6. Анализ результатов...")
    if yearly_errors_hist:
        best_index = np.argmin(yearly_errors_hist)
        best_solution = L_hist_yearly[best_index]
        best_error = yearly_errors_hist[best_index]
        best_hist = H_hist_yearly[best_index]
        
        print("\n" + "=" * 60)
        print("ЛУЧШЕЕ РЕШЕНИЕ")
        print("=" * 60)
        print(f"Ошибка по годам: {best_error:.6f}")
        print(f"Гистограмма: {best_hist}")
        
        print("\nСостав решения:")
        for i, line in enumerate(best_solution):
            rotation_sequence = Mcrops[line.m]
            rotation_str = "->".join([Crops[crop_id] for crop_id in rotation_sequence])
            print(f"  Строка {i + 1}: Севооборот {line.m} (фаза {line.fi})")
            print(f"            Последовательность: {rotation_str}")
        
        # Анализ распределения по годам для лучшего решения
        print("\nАнализ распределения по годам (первые 5 лет):")
        yearly_errors, total_error = calculate_yearly_error(best_solution)
        
        # Создаем матрицу культур по годам для анализа
        yearly_analysis = np.zeros((min(5, N2), len(Crops) + 1), dtype=float)
        for line in best_solution:
            m_index = line.m
            phase = line.fi
            rotation = Mcrops[m_index]
            rotation_length = len(rotation)
            
            for year in range(min(5, N2)):
                pos_in_rotation = (year + phase) % rotation_length
                crop_id = rotation[pos_in_rotation]
                yearly_analysis[year, crop_id] += 1
        
        # Нормализуем и выводим
        for year in range(min(5, N2)):
            total = np.sum(yearly_analysis[year, :])
            if total > 0:
                yearly_analysis[year, :] /= total
            
            print(f"\nГод {year + 1}:")
            for crop_id, crop_name in Crops.items():
                share = yearly_analysis[year, crop_id]
                if share > 0:
                    target = OptimalCropQuantity[crop_id]
                    deviation = share - target
                    print(f"  {crop_name}: {share:.3f} (цель: {target:.3f}, отклонение: {deviation:+.3f})")
            print(f"  Ошибка года: {yearly_errors[year]:.6f}")
        
        print(f"\nСуммарная ошибка по годам: {total_error:.6f}")
        
        # Сравнение с общей ошибкой
        matrix_error = calculate_error(best_solution)
        print(f"Общая ошибка (по матрице): {matrix_error:.6f}")
        
    else:
        print("Не удалось найти подходящие решения")
    
    print("\n" + "=" * 60)
    print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)

if (UserInput == 2 or UserInput ==12):
    # Параметры можно настроить
    
    INPUT_FILE = "matrix_optimization_results.txt"
    OUTPUT_FILE = "final_optimization_results.txt"
    
    final_optimization_stage(
        max_error=MAX_ERROR,
        input_filename=INPUT_FILE,
        output_filename=OUTPUT_FILE
    )
#print("\n--- Тестирование calculate_error ---")

#print("\n--- Тестирование расчета ошибки по годам ---")

# # Создаем тестовое решение
# test_solution = [
#     CropRotationLine(m=0, fi=0),  # [1,2]
#     CropRotationLine(m=1, fi=0),  # [1,3,4]
# ]

# yearly_errors, total_yearly_error = calculate_yearly_error(test_solution)

# print(f"Общая ошибка по годам: {total_yearly_error:.6f}")
# print(f"Ошибки по годам (первые 10 из {N2}):")
# for year in range(min(10, N2)):
#     print(f"  Год {year + 1}: {yearly_errors[year]:.6f}")

# # Сравним с общей ошибкой (по всей матрице)
# total_matrix_error = calculate_error(test_solution)
# print(f"Общая ошибка (по матрице): {total_matrix_error:.6f}")
# print(f"Отношение ошибок (годы/матрица): {total_yearly_error/total_matrix_error:.3f}")

# # Визуализация для лучшего понимания
# print("\n--- Анализ распределения по годам ---")
# print("Для первых 5 лет покажем распределение культур:")

# yearly_crop_matrix = np.zeros((min(5, N2), len(Crops) + 1), dtype=float)
# for line in test_solution:
#     m_index = line.m
#     phase = line.fi
#     rotation = Mcrops[m_index]
#     rotation_length = len(rotation)
    
#     for year in range(min(5, N2)):
#         pos_in_rotation = (year + phase) % rotation_length
#         crop_id = rotation[pos_in_rotation]
#         yearly_crop_matrix[year, crop_id] += 1

# # Нормализуем и выводим
# for year in range(min(5, N2)):
#     total = np.sum(yearly_crop_matrix[year, :])
#     if total > 0:
#         yearly_crop_matrix[year, :] /= total
    
#     print(f"\nГод {year + 1}:")
#     for crop_id, crop_name in Crops.items():
#         share = yearly_crop_matrix[year, crop_id]
#         if share > 0:
#             target = OptimalCropQuantity[crop_id]
#             print(f"  {crop_name}: {share:.3f} (цель: {target:.3f})")