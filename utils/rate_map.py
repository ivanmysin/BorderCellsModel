import numpy as np

def gaussian_kernel_2d(size, sigma):
    """Создает нормированное 2D гауссово ядро размера size x size."""
    kernel = np.zeros((size, size))
    center = size // 2
    total = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            value = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i, j] = value
            total += value
    return kernel / total  # Нормировка: сумма = 1

def convolve_2d(image, kernel):
    """Простая 2D свёртка с нулевым дополнением (zero padding)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result

def build_spatial_maps(
        activity: np.ndarray,
        positions: np.ndarray,
        bins: int = 50,
        range_xy: tuple = None):
    """
    Строит карту посещений, карту активности и нормированную карту пространственной
    активности нейрона (например, клетки места).

    Параметры
    ----------
    activity : np.ndarray, shape (T,)
        Сглаженная непрерывная активность нейрона в каждый момент времени.
    positions : np.ndarray, shape (T, 2)
        Координаты животного в те же моменты времени (x, y).
    bins : int or (int, int)
        Число бинов по осям. Если одно число, то одинаковое по x и y.
    range_xy : ((xmin, xmax), (ymin, ymax)) или None
        Диапазон координат для гистограммы. Если None, вычисляется по данным.
    plot : bool
        Если True, выводит нормированную карту на экран.
    title : str
        Заголовок для графика (используется только при plot=True).

    Возвращает
    ----------
    occupancy_map : np.ndarray, shape (bins_y, bins_x)
        Карта посещений (количество временных шагов в каждом бине).
    activity_map : np.ndarray, shape (bins_y, bins_x)
        Суммарная активность нейрона в каждом бине.
    rate_map : np.ndarray, shape (bins_y, bins_x)
        Нормированная карта: activity_map / occupancy_map.
    info : dict
        Поля: 'x_edges', 'y_edges' – границы бинов.
    """
    # Проверка формы входных данных
    if activity.ndim != 1 or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "activity должен быть одномерным массивом, positions – двумерным с двумя столбцами."
        )
    if len(activity) != positions.shape[0]:
        raise ValueError("Длины activity и positions должны совпадать.")

    # Определяем диапазон, если не задан
    if range_xy is None:
        xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
        ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
        range_xy = ((xmin, xmax), (ymin, ymax))

    # Карта посещений (вес каждого шага = 1)
    occupancy_map, x_edges, y_edges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins,
        range=range_xy
    )

    # Карта активности (вес = активность нейрона)
    activity_map, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins,
        range=range_xy,
        weights=activity
    )

    # Параметры фильтра
    kernel_size = 15
    sigma = 1.0

    # Создание нормированного гауссового ядра
    kernel = gaussian_kernel_2d(kernel_size, sigma)

    activity_map = convolve_2d(activity_map, kernel)
    occupancy_map = convolve_2d(occupancy_map, kernel)

    # Нормировка: делим активность на посещения, избегая деления на 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.divide(activity_map, occupancy_map)
        rate_map[occupancy_map == 0] = np.nan   # не посещённые бины — NaN

    info = {'x_edges': x_edges, 'y_edges': y_edges}
    return rate_map, occupancy_map, activity_map, info

