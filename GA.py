import numpy as np, random, pandas as pd
from typing import List, Dict

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

    def __repr__(self):
        # Defines the printable representation of the City
        return "(" + str(self.x) + "," + str(self.y) + ")"

# Houston = City(1, 2.5)
# Boston = City(20.2, 12)
# print(Houston)
# print(type(Houston))
# print(Houston.distance(Boston))

class Route:
    def __init__(self, route: List['City']) -> None:
        # init defines a route between cities
        self.route = route
        self.distance: float = 0
        self.fitness: float = 0.0
        self.numStops: int = len(self.route)

    def routeDistance(self) -> float:
        if self.distance == 0: # prevents from recalculating fitness for a route
            pathDistance = 0.0 # temporary calculation variable
            for i in range(self.numStops):
                fromCity = self.route[i]
                # if you are not at the last city, the next city is the next
                # in the route. Else, it is the first city you must go back to.
                if i + 1 < self.numStops:
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self) -> float:
        if self.fitness == 0:
            self.fitness = 1 / self.routeDistance()
        return self.fitness

# for generating random lists of cities
def initialPopulation(popSize: int, numCities: int):
    # creates a list of random cities with k entries
    cityList = []
    population = []
    for i in range(numCities):
        cityList.append(City(x=round(random.random()*200), y=round(random.random()*200)))
    for i in range(popSize):
        population.append(random.sample(cityList, len(cityList)))
    return population

# cityList = []
# for i in range(8):
#     cityList.append(City(x=round(random.random()*200), y=round(random.random()*200)))
# R1 = Route(cityList)
# print(R1.routeFitness())
# print(R1.distance)
# print(cityList)

# abc = initialPopulation(5, 2)
# print(abc)

def rankRoutes(population: List[List[City]]):
    fitnessResults: Dict = {}
    for i in range(len(population)):
        fitnessResults[i] = Route(population[i]).routeFitness() # dict of each
        #### unfinished ####
