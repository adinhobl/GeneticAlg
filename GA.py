import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Any

# From: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35


class City:
    def __init__(self, x: float, y: float) -> None:
        # init defines a single city's location using x and y coordinates
        self.x = x
        self.y = y

    def distance(self, city: 'City') -> float:
        # Calculates the distance between City and another city
        xDist = abs(self.x - city.x)
        yDist = abs(self.y - city.y)
        distance = np.sqrt(xDist ** 2 + yDist ** 2)
        return distance

    def to_tupl(self) -> Tuple:
        return (self.x, self.y)

    def __repr__(self) -> str:
        # Defines the printable representation of the City
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Route:
    def __init__(self, route: List['City']) -> None:
        # init defines a route between cities
        self.route = route
        self.distance: float = 0.0
        self.fitness: float = 0.0
        self.numStops: int = len(self.route)

    def routeDistance(self) -> float:
        # calculates the total distance of a route
        if self.distance == 0:  # prevents from recalculating fitness for a route
            pathDistance = 0.0  # temporary calculation variable
            for i in range(self.numStops):
                fromCity = self.route[i]
                # if you are not at the last city, the next city is the next
                # in the route. Else, you must go back to the first city.
                if i + 1 < self.numStops:
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self) -> float:
        # calculates the fitness of a route from its distance
        if self.fitness == 0:
            self.fitness = calcFitness(self.routeDistance())
        return self.fitness

    def coordinates(self) -> Tuple[List[float], List[float]]:
        x_list, y_list = [], []
        for city in self.route:
            x_list.append(city.to_tupl()[0])
            y_list.append(city.to_tupl()[1])
        return x_list, y_list

    def __repr__(self) -> Any:
        # Defines the printable representation of the City
        return str(self.route)


def calcFitness(routeDistance: float):
    return 1 / routeDistance

# for generating random lists of cities


def initialPopulation(popSize: int, numCities: int, cityListIn: List = None) -> List[List['City']]:
    # creates a list of random cities with k entries or use cityListIn, if provided
    # note that if you use cityListIn, you still must provide its numCities and popSize
    cityList: List = []

    if cityListIn != None:
        for city in cityListIn:
            cityList.append(city)
    else:
        for i in range(numCities):
            cityList.append(City(x=round(random.random()*200),
                                 y=round(random.random()*200)))

    population = []
    for i in range(popSize):
        if cityListIn != None:
            random.seed(11)
        population.append(random.sample(cityList, len(cityList)))
    return population


def rankRoutes(population: List[List['City']]) -> List[Tuple[int, float]]:
    # ranks the routes in a list of routes according to fitness
    fitnessResults: Dict = {}
    for i in range(len(population)):
        # makes a list of cities into a route, then finds fitness
        fitnessResults[i] = Route(population[i]).routeFitness()
    # can also use itemgetter(2)
    return sorted(fitnessResults.items(), key=lambda x: x[1], reverse=True)


def selection(popRanked: List[Tuple[int, float]], numElites: int = 0) -> List[int]:
    # select which individuals are saved as parents of the next generation
    # Fitness propporionate selection implemented with pd.sample
    # Eliteness implemented by picking top individuals

    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df["weights"] = 100 * df.Fitness / df.Fitness.sum()
    selection_results = df.sample(n=len(popRanked)-numElites,
                                  replace=True,
                                  weights=df.weights
                                  ).values[:, 0]
    elite_results = df.iloc[0:numElites, 0].values
    # print(df) # to see the dataframe for checking
    selection_results = list(map(int, np.concatenate(
        (selection_results, elite_results)).tolist()))
    return selection_results


def matingPool(population: List[List['City']], selection_results: List[int]) -> List[List['City']]:
    # picks the mating individuals out of the population based on their selection_results
    mating_pool: List = []
    for i in range(len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool


def breed(parent1: List['City'], parent2: List['City']) -> List['City']:
    # uses ordered crossover to breed 2 parents to make a new individual
    # print("Parent1: ", parent1, "\n")
    # print("Parent2: ", parent2, "\n")
    child: List = [None] * (len(parent1))
    # print("Child initialization: ", child, "\n")

    geneFromParent1 = (random.randint(0, len(child) - 1),
                       random.randint(0, len(child) - 1))
    geneFromParent1_start = min(geneFromParent1)
    geneFromParent1_end = max(geneFromParent1)
    # print(geneFromParent1, geneFromParent1_start, geneFromParent1_end, "\n")

    for gene in range(geneFromParent1_start, geneFromParent1_end + 1):
        child[gene] = parent1[gene]

    # print("Child after p1: ", child, "\n")

    for i in range(0, len(child)):
        for j in parent2:
            if j not in child:
                if child[i] == None:
                    child[i] = j

    # print("Child after p2: ", child, "\n")
    return child


def breedPopulation(mating_pool: List[List['City']], numElites: int = 0):
    children: List = []  # final list of children
    numNonElite = len(mating_pool) - numElites
    pool = random.sample(mating_pool, len(mating_pool)
                         )  # shuffles the pool around

    # Carry elites over to next breeding population
    for i in range(1, numElites+1):
        children.append(mating_pool[-i])

    # breed population - numElites number of individuals mate with the elites.
    for i in range(0, numNonElite):
        child = breed(pool[i], pool[len(mating_pool)-i-1])
        children.append(child)

    return children


def swapMutation(individual: List['City'], mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            individual[swapped], individual[swapWith] = \
                individual[swapWith], individual[swapped]

    return individual


def mutatePopulation(children: List[List['City']], mutationRate=0):
    mutatedPop: List = []

    for individual in range(0, len(children)):
        mutatedIndividual = swapMutation(children[individual], mutationRate)
        mutatedPop.append(mutatedIndividual)

    return mutatedPop


def nextGeneration(currentGen: List[List['City']], numElites: int, mutationRate: float = 0):
    popRanked = rankRoutes(currentGen)

    # extracting the best route of this generation
    bestCurrentGenRoute = Route(currentGen[popRanked[0][0]])
    bestCurrentGenFitness = bestCurrentGenRoute.routeFitness()
    bestCurrentGenDistance = bestCurrentGenRoute.routeDistance()

    selectionResults = selection(popRanked, numElites)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, numElites)
    nextGeneration = mutatePopulation(children, mutationRate)

    return nextGeneration, bestCurrentGenRoute, bestCurrentGenFitness, bestCurrentGenDistance


def geneticAlgorithm(popSize: int, numCities: int, numElites: int, numGens: int, mutationRate: float = 0.01, cityListIn: List = None):
    pop = initialPopulation(popSize, numCities, cityListIn)
    bestInitialRoute = Route(pop[rankRoutes(pop)[0][0]])
    print("Initial Distance: " + str(bestInitialRoute.routeDistance()))

    bestRouteByGen: List = []
    bestFitnessByGen: List = []
    bestDistanceByGen: List = []

    for i in range(0, numGens):
        pop, bestCurrentGenRoute, bestCurrentGenFitness, bestCurrentGenDistance = \
            nextGeneration(pop, numElites, mutationRate)

        bestRouteByGen.append(bestCurrentGenRoute)
        bestFitnessByGen.append(bestCurrentGenFitness)
        bestDistanceByGen.append(bestCurrentGenDistance)

        # used for testing convergence
        # if bestCurrentGenDistance < 852:
        #     print(i)
        #     break

    bestFinalRoute = Route(pop[rankRoutes(pop)[0][0]])
    print("Final Distance: " + str(bestFinalRoute.routeDistance()))

    params = [popSize, numCities, numElites, numGens, mutationRate, cityListIn]

    return bestFinalRoute, bestRouteByGen, bestFitnessByGen, bestDistanceByGen, params


def distancePlot(bestDistanceByGen: List[int], params: List):
    plt.plot(bestDistanceByGen)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    s = "popSize: " + str(params[0]) + "\nnumCities: " + str(params[1]) + \
        "\nnumGens: " + str(params[3]) + "\nmutationRate: " + str(params[4])
    plt.text(330, 2010, s)
    plt.text(0, bestDistanceByGen[0], bestDistanceByGen[0].round(1))
    plt.text(len(bestDistanceByGen),
             bestDistanceByGen[-1], bestDistanceByGen[-1].round(1))
    plt.show()


def evolutionPlot(bestRouteByGen, bestDistanceByGen, cityListIn):

    fig, ax = plt.subplots()
    xdata = []
    ydata = []
    line, = plt.plot([], [], 'C3', animated=True)
    gen_text = ax.text(150, 185, '')

    for i in range(len(bestRouteByGen)):
        x, y = bestRouteByGen[i].coordinates()
        xdata.append(x)
        ydata.append(y)

    def init():
        x = [i[0] for i in cityListIn]
        y = [i[1] for i in cityListIn]
        ax.scatter(x, y, s=60)
        gen_text.set_text('')

        return line,

    def animate(i):
        numGens = len(bestRouteByGen)
        line.set_data(xdata[i % numGens], ydata[i % numGens])

        gen_text.set_text("Generation:" + str(i % numGens) +
                          "\nDistance: " + str(round(bestDistanceByGen[i % numGens], 2)))
        return line,  gen_text

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, blit=True, repeat_delay=2000, interval=50, save_count=500)

    # ani.save("GA4TSM.gif")
    plt.show()
