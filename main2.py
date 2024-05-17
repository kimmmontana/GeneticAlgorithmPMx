import random
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from constants import (
    POOL_SIZE,
    MUTATION_RATE,
    CROSSOVER_RATE,
    CROSS_SECTION_1,
    CROSS_SECTION_2,
    MARGIN
)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"


class City:
    def __init__(self, name, coordinate, representation_number):
        self.name = name
        self.coordinate = Point(*coordinate)
        self.representation_number = representation_number

    def __repr__(self):
        return f"City(name={self.name}, coordinate={self.coordinate}, representation_number={self.representation_number})"


cities = [
    City("Muntinlupa", (1, 1), 1),
    City("San Juan", (5, 8), 2),
    City("Paranaque", (2, 3), 3),
    City("Makati", (4, 6), 4),
    City("Taguig", (5, 3), 5),
    City("Las Pinas", (1, 2), 6),
    City("Pasig", (6, 7), 7),
    City("Manila", (4, 4), 8),
    City("Pasay", (3, 3), 9),
    City("Valenzuela", (7, 9), 10),
    City("Mandaluyong", (5, 7), 11),
    City("Marikina", (7, 8), 12),
    City("Malabon", (8, 9), 13),
    City("Caloocan", (8, 8), 14),
    City("Quezon", (7, 7), 15),
]


def map_city(individual, generation, fitness):
    plt.cla()
    x_coords = [cities[city - 1].coordinate.x for city in individual]
    y_coords = [cities[city - 1].coordinate.y for city in individual]

    # plt.figure(figsize=(10, 8))
    plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], marker="o")

    for i, city in enumerate(individual):
        if i == 0:
            plt.plot(
                cities[city - 1].coordinate.x,
                cities[city - 1].coordinate.y,
                marker="o",
                markersize=10,
                color="red",
            )
        elif i == len(cities) - 1:
            plt.plot(
                cities[city - 1].coordinate.x,
                cities[city - 1].coordinate.y,
                marker="o",
                markersize=10,
                color="green",
            )
        else:
            plt.plot(
                cities[city - 1].coordinate.x,
                cities[city - 1].coordinate.y,
                marker="o",
                markersize=6,
                color="blue",
            )
        plt.text(
            cities[city - 1].coordinate.x,
            cities[city - 1].coordinate.y,
            cities[city - 1].name,
            fontsize=9,
            ha="right",
        )

    plt.title(f"Generation {generation + 1}")
    plt.text(0, 0, fitness)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.pause(0.01)


def map_line(graph):
    plt.plot(graph, marker="o", linestyle="-", markersize="3", color="blue")
    plt.grid(True)
    plt.show()


def distance_formula(city1: City, city2: City):
    return np.sqrt(
        (city1.coordinate.x - city2.coordinate.x) ** 2
        + (city1.coordinate.y - city2.coordinate.y) ** 2
    )


def initialize_pool():
    print(f"Generating {POOL_SIZE} individuals")
    pool = []
    unique_individuals = set()
    for _ in range(POOL_SIZE):
        individual = list(range(1, len(cities) + 1))
        while tuple(individual) in unique_individuals:
            random.shuffle(individual)
        unique_individuals.add(tuple(individual))
        pool.append(individual)
    return pool

def select_parent(pool: list, best_fitness):
    if best_fitness == 0:
        chosen_parent = random.choice(pool)
        pool.remove(chosen_parent)
        return random.choice(pool)

    while pool:
        chosen_parent = random.choice(pool)
        parent_fitness = evaluate_fitness(chosen_parent)
        pool.remove(chosen_parent)
        # print(f"pool remaining: {len(pool)}: {parent_fitness} < {best_fitness + MARGIN}?")
        if parent_fitness < best_fitness + MARGIN:
            return chosen_parent
    return []

def evaluate_fitness(individual):
    if not individual:
        return 0
    total_distance = 0
    for i in range(len(individual) - 1):
        from_city = individual[i] - 1
        to_city = individual[i + 1] - 1
        total_distance += distance_matrix[from_city][to_city]
    total_distance += distance_matrix[individual[-1] - 1][individual[0] - 1]
    return total_distance


def pmx(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2

    size = len(parent1)
    p1, p2 = [0] * size, [0] * size

    for k in range(size):
        p1[parent1[k] - 1] = k
        p2[parent2[k] - 1] = k

    for k in range(CROSS_SECTION_1, CROSS_SECTION_2):
        temp1 = parent1[k]
        temp2 = parent2[k]
        parent1[k], parent1[p1[temp2 - 1]] = temp2, temp1
        parent2[k], parent2[p2[temp1 - 1]] = temp1, temp2
        p1[temp1 - 1], p1[temp2 - 1] = p1[temp2 - 1], p1[temp1 - 1]
        p2[temp1 - 1], p2[temp2 - 1] = p2[temp2 - 1], p2[temp1 - 1]

    return parent1, parent2


def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def linear_check(parent1, parent2):
    # print(zip(parent1, parent2))
    for i in zip(parent1, parent2):
        if i[0] == i[1]:
            return True
    return False

def genetic_algorithm(pool):
    best_individual = select_parent(pool, 0)
    best_fitness = evaluate_fitness(best_individual)
    fitness_graph = []
    plt.ion()
    generation = 0
    while len(pool) > 1:
        second_parent = select_parent(pool, best_fitness)
        
        if second_parent == []:
            break

        if linear_check(best_individual, second_parent):
            # pool.append(second_parent)
            continue

        offspring1, offspring2 = pmx(best_individual[:], second_parent)

        print(
            f"Looking for best offspring in Generation {generation + 1}: Parent 1: {best_individual}, Parent 2: {second_parent}, Offspring 1: {offspring1}, Offspring 2: {offspring2}"
        )
        current_individual = min([best_individual, offspring1, offspring2], key=evaluate_fitness)
        if current_individual == best_individual:
            continue

        best_individual = current_individual
        best_fitness = evaluate_fitness(best_individual)

        generation += 1
        fitness_graph.append(best_fitness)
        print(
            f"Generation {generation}: Best Fitness = {best_fitness}, Best Route = {best_individual}, Total Distance = {best_fitness}"
        )
        map_city(best_individual, generation-1, best_fitness)
        sleep(1)
    plt.ioff()

    print(f"\nBest route after {generation} generations: {best_individual}")
    print(f"Best route distance: {best_fitness}")
    plt.show()
    return best_individual, best_fitness, fitness_graph


num_cities = len(cities)
distance_matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distance_matrix[i][j] = distance_formula(cities[i], cities[j])

pool = initialize_pool()
best_individual, best_fitness, fitness_graph = genetic_algorithm(pool)

map_line(fitness_graph)
