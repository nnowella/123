import numpy as np

# =====================================================
# 1. Создание массивов и атрибуты
# =====================================================
print("=" * 50)
print("1. Создание массивов и атрибуты")
print("=" * 50)

a1 = np.array([2, 4, 6, 8, 10])
zeros = np.zeros(12)
ones = np.ones(9)
fives = np.full(6, 5)
seq = np.arange(12)            # от 0 до 11
seq_step = np.arange(2, 31, 4)  # от 2 до 30 с шагом 4
lin = np.linspace(0, 1, 5)     # 5 равномерных чисел от 0 до 1

print("Массив:", a1)
print("12 нулей:", zeros)
print("9 единиц:", ones)
print("6 пятёрок:", fives)
print("От 0 до 11:", seq)
print("От 2 до 30 с шагом 4:", seq_step)
print("5 чисел от 0 до 1:", lin)

print("\nАтрибуты массива", a1)
print("Размер:", a1.size)
print("Форма:", a1.shape)
print("Тип данных:", a1.dtype)
print("Количество измерений:", a1.ndim)

m43 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]])
print("\nМассив 4x3:\n", m43)
print("Форма:", m43.shape, "Размер:", m43.size, "Тип:", m43.dtype)

m35 = np.arange(1, 16).reshape(3, 5)
print("\nМатрица 3x5:\n", m35)

m52 = np.ones((5, 2))
row = m52.reshape(1, 10)
col = m52.reshape(10, 1)
print("\nВектор-строка, форма:", row.shape)
print("Вектор-столбец, форма:", col.shape)

# =====================================================
# 2. Индексация и срезы
# =====================================================
print("\n" + "=" * 50)
print("2. Индексация и срезы")
print("=" * 50)

arr = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30])

print("Первый элемент:", arr[0])
print("Последний элемент:", arr[-1])
print("Элемент с индексом 3:", arr[3])
print("Элементы с индексами 1, 4, 6:", arr[[1, 4, 6]])

print("\nПервые 5 элементов:", arr[:5])
print("Со 2-го по 6-й включительно:", arr[1:6])
print("Каждый третий элемент:", arr[::3])
print("В обратном порядке:", arr[::-1])
print("Последние 4 элемента:", arr[-4:])

print("\nБольше 15:", arr[arr > 15])
print("Больше 9 и меньше 27:", arr[(arr > 9) & (arr < 27)])

matrix = np.array([[2, 4, 6, 8],
                   [10, 12, 14, 16],
                   [18, 20, 22, 24]])

print("\nПервая строка:", matrix[0])
print("Третий столбец:", matrix[:, 2])
print("Элемент во 2-й строке и 3-м столбце:", matrix[1, 2])
print("Первые 2 строки, последние 2 столбца:\n", matrix[:2, -2:])

# =====================================================
# 3. Арифметические операции и агрегаты
# =====================================================
print("\n" + "=" * 50)
print("3. Арифметические операции и агрегаты")
print("=" * 50)

a = np.array([6, 9, 12, 15, 18, 21])

print("a + 7:", a + 7)
print("a * 3:", a * 3)
print("a ** 2:", a ** 2)
print("a % 4:", a % 4)

print("\nСумма:", a.sum())
print("Произведение:", a.prod())
print("Минимум:", a.min(), "Максимум:", a.max())
print("Среднее:", a.mean())
print("Медиана:", np.median(a))
print("Стандартное отклонение:", round(a.std(), 2))

b = np.array([1, 2, 3, 4, 5, 6])

print("\na + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", np.round(a / b, 2))

i_min, i_max = a.argmin(), a.argmax()
print("\nИндекс минимума:", i_min, "значение:", a[i_min])
print("Индекс максимума:", i_max, "значение:", a[i_max])

mask = a > 10
print("\nСумма элементов > 10:", a[mask].sum())
print("Количество элементов > 10:", mask.sum())

# =====================================================
# 4. Двумерные массивы и агрегаты по осям
# =====================================================
print("\n" + "=" * 50)
print("4. Двумерные массивы и агрегаты по осям")
print("=" * 50)

scores = np.array([[4, 5, 3, 4],   # студент 1
                   [5, 5, 4, 3],   # студент 2
                   [3, 4, 4, 5],   # студент 3
                   [4, 3, 5, 5]])  # студент 4

print("Форма:", scores.shape, "Размер:", scores.size)
print(scores)

print("\nСумма по каждому студенту:", scores.sum(axis=1))
print("Сумма по каждому предмету:", scores.sum(axis=0))

mean_students = np.round(scores.mean(axis=1), 2)
mean_subjects = np.round(scores.mean(axis=0), 2)
print("\nСредний балл студентов:", mean_students)
print("Средний балл предметов:", mean_subjects)

print("\nМаксимальный балл по каждому студенту:", scores.max(axis=1))
print("Минимальный балл по каждому предмету:", scores.min(axis=0))

best = scores.mean(axis=1).argmax()
print("\nЛучший студент: индекс", best,
      "(студент", best + 1, "), средний балл:", mean_students[best])

print("\n+1 балл, но не больше 5:\n", np.minimum(scores + 1, 5))

# =====================================================
# 5. Изменение формы, объединение, сортировка
# =====================================================
print("\n" + "=" * 50)
print("5. Изменение формы, объединение, сортировка")
print("=" * 50)

r = np.arange(1, 13)
for shape in [(3, 4), (2, 6), (4, 3)]:
    res = r.reshape(shape)
    print(f"Матрица {shape[0]}x{shape[1]}, форма {res.shape}:\n{res}\n")

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("Один вектор:", np.concatenate([x, y]))
print("Матрица 2x3:\n", np.vstack([x, y]))
print("Матрица 3x2:\n", np.column_stack([x, y]))

data = np.array([10, 3, 7, 1, 9, 4, 8])
print("\nПо возрастанию:", np.sort(data))
print("По убыванию:", np.sort(data)[::-1])

m = np.array([[2, 7, 3],
              [6, 1, 8],
              [4, 9, 5]])

print("\nСуммы по строкам:", m.sum(axis=1))
print("Суммы по столбцам:", m.sum(axis=0))
print("Максимум по матрице:", m.max())
row_i, col_i = np.unravel_index(m.argmax(), m.shape)
print("Индекс максимума: строка", int(row_i), "столбец", int(col_i))

# =====================================================
# 6. Крестики-нолики
# =====================================================
print("\n" + "=" * 50)
print("6. Крестики-нолики")
print("=" * 50)


def winner(board):
    # Все линии: 3 строки, 3 столбца, 2 диагонали
    lines = list(board) + list(board.T) + [np.diag(board),
                                            np.diag(np.fliplr(board))]
    for line in lines:
        if np.all(line == 1):
            return "Победили крестики"
        if np.all(line == 0):
            return "Победили нолики"
    return "Ничья"


board = np.array([[1, 0, 1],
                  [-1, 1, 0],
                  [0, -1, 1]])

print(board)
print(winner(board))
