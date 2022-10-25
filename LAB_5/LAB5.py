import numpy
import numpy as np
from scipy import spatial
from collections import Counter
import random

np.random.seed(42)
random.seed(42)


class Individual:
    def __init__(self, gnome):
        self.gnome = gnome
        self.fitness = 0

    def count_fitness(self, dist_matrix):
        s = 0
        for i in range(1, len(self.gnome)):
            s += dist_matrix[self.gnome[i]][self.gnome[i-1]]
        self.fitness = s

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


def exchange(gen1, gen2):
    delimetr = random.randint(0, len(gen1) - 1)
    new_genom = gen1[:delimetr]
    for i in range(delimetr, len(gen2)):
        if gen2[i] not in new_genom:
            new_genom.append(gen2[i])
    if len(new_genom) != len(gen2):
        for i in range(delimetr, len(gen1)):
            if gen1[i] not in new_genom:
                new_genom.append(gen1[i])
        return Individual(new_genom)
    return Individual(new_genom)


def crossover(individ1, individ2):
    genom1, genom2 = (individ1.gnome, individ2.gnome)
    return exchange(genom1, genom2), exchange(genom2, genom1)


# Создание стартовой популяции
def create_start_population(d_matrix, num_individ):
    init_ways = [[i for i in range(len(d_matrix[0]))] for j in range(num_individ)]
    population = []
    for i in range(len(init_ways)):
        random.shuffle(init_ways[i])
        population.append(Individual(init_ways[i]))
        population[i].count_fitness(d_matrix)
    return population

def mutation(genom):
    pos1 = 0
    pos2 = 0
    while pos1 == pos2:
        pos1, pos2 = (random.randint(0, len(genom)-1), random.randint(0, len(genom)-1))
    genom[pos1], genom[pos2] = genom[pos2], genom[pos1]
    return genom


def fitness_meaning_in_populatoin(population):
    return [population[i].fitness for i in range(len(population))]


def genetic_algo(d_matrix, temperature=100000, num_ind=35, cross_chance=0.5, mutation_chance=0.05):
    pop = create_start_population(d_matrix, num_ind)
    while temperature > 0:
        fitness_in_pop = fitness_meaning_in_populatoin(pop)
        choice_weights = [1 / fitness_in_pop[i] for i in range(len(fitness_in_pop))]
        choice_weights = [choice_weights[i] / sum(choice_weights) for i in range(len(fitness_in_pop))]
        selected_genoms = np.random.choice(pop, size=2, p=choice_weights)
        #Выполняем скрещивание
        cross = crossover(selected_genoms[0], selected_genoms[1])
        for i in range(len(cross)):
            cross[i].count_fitness(d_matrix)

        #Выполняем мутации
        for i in range(len(pop)):
            if random.random() < mutation_chance:
                pop[i].gnome = mutation(pop[i].gnome)

        #Добавление новой особи, вместо худших особей в популяции
        pos1 = np.argmax(fitness_in_pop)
        fitness_in_pop[pos1] = 0
        pos2 = np.argmax(fitness_in_pop)
        pop[pos1] = cross[0]
        pop[pos2] = cross[1]

        temperature -= 1

    return pop[np.argmin(fitness_meaning_in_populatoin(pop))]


if __name__ == "__main__":
    num_points = 20
    points_coordinate = np.random.randint(100, size=(num_points, 2))
    # print("Координаты вершин:\n", points_coordinate[:10], "\n")

    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print("Матрица расстояний:\n", distance_matrix)

    print(genetic_algo(distance_matrix).fitness)
    print("""""")