import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points]][routine[(i + 1) % num_points]] for i in range(num_points)])


class ACA:
    def __init__(self, func, n_dim, size_pop, alpha, beta, per, distance_matrix=None, max_iter=10000):
        self.tabu_set = None
        self.func = func
        self.generation_best_distance = []
        self.generation_best_way = []
        self.n_dim = n_dim  # Кол-во городов
        self.size_pop = size_pop  # Размер популяции муравьев
        self.max_iter = max_iter  # Максимальное кол-во шагов
        self.alpha = alpha  # Коэффициент важности феромона при выборе пути
        self.beta = beta  # Важность измерения важности обратного расстояния для грани
        self.per = per  # Скорость испарения феромонов

        self.distance_matrix = np.array(distance_matrix)  # Матрицы расстояний между городами
        self.pher_matrix = np.ones((self.n_dim, self.n_dim))  # Матрица феромонов
        self.road_matrix = np.zeros((size_pop, n_dim)).astype(int)  # Путь каждого муравья
        self.distance = 0  # Дистанция пути муравья внутри поколения
        self.best_way, self.best_distance = [], []

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        start_arr = np.linspace(0, self.n_dim-1, num=self.size_pop, dtype=int)
        for i in range(self.max_iter):
            transition_matrix = self.pher_matrix ** self.alpha * (1 / self.distance_matrix) ** self.beta
            for j in range(self.size_pop):  # Проход по популяции
                self.road_matrix[j][0] = start_arr[j]
                self.tabu_set = []  # Вершины, которые муравей уже посетил
                for k in range(self.n_dim - 1):  # Проход одного муравья по графу
                    self.tabu_set.append(self.road_matrix[j][k])
                    allow_list = list(set(range(0, self.n_dim)) - set(self.tabu_set))
                    probability = transition_matrix[self.road_matrix[j][k], allow_list]
                    probability = probability/probability.sum()
                    next_point = np.random.choice(allow_list, size=1, p=probability)
                    self.road_matrix[j][k+1] = next_point
            road_distance = np.array([self.func(i) for i in self.road_matrix])
            index_best = road_distance.argmin()
            self.generation_best_distance.append(road_distance[index_best])
            self.generation_best_way.append(self.road_matrix[index_best])

            for start in range(len(self.road_matrix)):
                for stop in range(len(self.road_matrix[start])-1):
                    self.pher_matrix[self.road_matrix[start][stop]][self.road_matrix[start][stop+1]] += \
                        1/distance_matrix[self.road_matrix[start][stop]][self.road_matrix[start][stop+1]]
        min_distance_index = 0
        for i in range(len(self.generation_best_distance)):
            if self.generation_best_distance[i] < self.generation_best_distance[min_distance_index]:
                min_distance_index = i
        self.best_distance = self.generation_best_distance[min_distance_index]
        self.best_way = self.generation_best_way[min_distance_index]
        return self.best_way, self.best_distance

def main():
    # создание объекта алгоритма муравьиной колонии
    aca = ACA(cal_total_distance, num_points, 20, 15, 2, 0.98, distance_matrix)
    best_x, best_y = aca.run(20)
    print("Наилучший путь: ", best_x, "\n", "Наилучшее расстояние: ", best_y)
    # Вывод результатов на экран
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    for index in range(0, len(best_points_)):
        ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
    ax[0].plot(best_points_coordinate[:, 0],
               best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.generation_best_way).cummin().plot(ax=ax[1])
    # изменение размера графиков
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.show()

if __name__ == "__main__":
    num_points = 20
    points_coordinate = np.random.randint(100, size=(num_points, 2))
    print("Координаты вершин:\n", points_coordinate[:10], "\n")

    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print("Матрица расстояний:\n", distance_matrix)
    start_time = time.time()
    main()
    print("time of execution: %s seconds" %abs (time.time() - start_time))