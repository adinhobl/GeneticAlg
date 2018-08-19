import numpy as np, random, pandas as pd
from typing import List, Dict, Tuple

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

    def __repr__(self) -> str:
        # Defines the printable representation of the City
        return "(" + str(self.x) + "," + str(self.y) + ")"

## Test for City class
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
        # calculates the total distance of a route
        if self.distance == 0: # prevents from recalculating fitness for a route
            pathDistance = 0.0 # temporary calculation variable
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
        #calculates the fitness of a route from its distance
        if self.fitness == 0:
            self.fitness = 1 / self.routeDistance()
        return self.fitness

# for generating random lists of cities
def initialPopulation(popSize: int, numCities: int) -> List:
    # creates a list of random cities with k entries
    cityList = []
    population = []
    for i in range(numCities):
        cityList.append(City(x=round(random.random()*200), y=round(random.random()*200)))
    for i in range(popSize):
        population.append(random.sample(cityList, len(cityList)))
    return population

## Generic list of cities
# cityList = []
# for i in range(8):
#     cityList.append(City(x=round(random.random()*200), y=round(random.random()*200)))
# R1 = Route(cityList)
# print(R1.routeFitness())
# print(R1.distance)
# print(cityList)

## Test for initialPopulation
# abc = initialPopulation(10, 60)
# print(abc,"\n")

def rankRoutes(population: List[List['City']]) -> List[Tuple[int, float]]:
    # ranks the routes in a list of routes according to fitness
    fitnessResults: Dict = {}
    for i in range(len(population)):
        fitnessResults[i] = Route(population[i]).routeFitness() # makes a list of cities into a route, then finds fitness
    return sorted(fitnessResults.items(), key=lambda x: x[1], reverse=True) # can also use itemgetter(2)

## Test for rankRoutes - must have test for initialPopulation active
# cba = rankRoutes(abc)
# print(cba)

def selection(popRanked: List[Tuple[int, float]], numElites: int = 0) -> List[int]:
    # select which individuals are saved as parents of the next generation
    # Fitness propporionate selection implemented with pd.sample
    # Eliteness implemented by picking top individuals

    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df["weights"] = 100 * df.Fitness / df.Fitness.sum()
    selection_results = df.sample(n=len(popRanked)-numElites,
                                 replace=True,
                                 weights=df.weights
                                 ).values[:,0]
    elite_results = df.iloc[0:numElites,0].values
    # print(df) # to see the dataframe for checking
    selection_results = list(map(int,np.concatenate((selection_results, elite_results)).tolist()))
    return selection_results

## Test for selection - must have test for rankRoutes in place
# j = selection(cba,2)
# print(j)
