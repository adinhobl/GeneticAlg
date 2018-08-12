import numpy as np, random, pandas as pd


class City:
    def __init__(self, x: float, y: float) -> None:
        # init class defines a single city's location using x and y coordinates
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
