from GA import *


###Test for City class###

# Houston = City(1, 2.5)
# Boston = City(20.2, 12)
# print(Houston)
# print(type(Houston))
# print(Houston.distance(Boston))


###Generic list of cities###

# cityList = []
# for i in range(8):
#     cityList.append(City(x=round(random.random()*200), y=round(random.random()*200)))
# R1 = Route(cityList)
# print(R1.routeFitness())
# print(R1.distance)
# print(cityList)


###Test for initialPopulation###

# abc = initialPopulation(1, 25)
# print(abc, "\n")


#Test for rankRoutes###

# abc = initialPopulation(5, 20)
# print(abc,"\n")
# cba = rankRoutes(abc)
# print(cba)
# print("\n")


###Test for selection###

# abc = initialPopulation(5, 20)
# print(abc,"\n")
# cba = rankRoutes(abc)
# print(cba)
# numElites = 2
# j = selection(cba,numElites)
# print(j)
# print("\n")


###Test for matingPool###

# abc = initialPopulation(6, 20)
# print(abc, "\n")
# cba = rankRoutes(abc)
# print(cba)
# numElites = 2
# j = selection(cba, numElites)
# print(j)
# # pool = matingPool(abc, j)
# print(pool)
# print("\n")


###Test for breed###

# abc = initialPopulation(6, 10)
# #print(abc, "\n")
# cba = rankRoutes(abc)
# # print(cba)
# numElites = 2
# j = selection(cba,numElites)
# # print(j)
# pool = matingPool(abc, j)
# # print(pool)
# breeded = breed(pool[0], pool[1])
# print(breeded, '\n')


###Test for breedPopulation###

# abc = initialPopulation(6, 10)
# #print(abc, "\n")
# cba = rankRoutes(abc)
# # print(cba)
# numElites = 2
# j = selection(cba, numElites)
# # print(j)
# pool = matingPool(abc, j)
# # print(pool)
# next_gen = breedPopulation(pool, numElites)
# for individual in next_gen:
#     print(individual, ' ')


###Test for swapMutation###

# abc = initialPopulation(1, 2)[0]
# print(abc, "\n")
# for i in range(20):
#     cba = swapMutation(abc, 0.1)
#     print(cba)


###Test for mutatePopulation###

# abc = initialPopulation(2, 5)
# print(abc, "\n")
# for i in range(3):
#     cba = mutatePopulation(abc, 0.1)
#     print(cba)


###Test for nextGeneration###
# Note: it's hard to tell if this works with only this test

# abc = initialPopulation(3, 5)
# print(abc, "\n")
# for i in range(5):
#     abc = nextGeneration(abc, 1, 0.1)
#     print(abc)


###Test for geneticAlgorithm, random cityList###
# 100 routes, 25 cities, 20 Elites, 500 generations, 1% mutation rate

# geneticAlgorithm(100, 25, 20, 500, .01)

###Test for geneticAlgorithm, custom cityListIn###
# 100 routes, 25 cities, 20 Elites, 500 generations, 1% mutation rate

numCities: int = 25

# # Comment out this section
# cityList: List = []
# for i in range(numCities):
#     cityList.append(City(x=round(random.random()*200),
#                          y=round(random.random()*200)))

# Or comment out this section
cities = [(182, 19), (13, 170), (26, 9), (161, 39), (33, 103), (117, 85),
          (182, 158), (196, 22), (99, 159), (8, 23), (146, 33), (125, 185),
          (34, 100), (156, 67), (185, 184), (74, 57), (178, 169), (22, 199),
          (44, 47), (140, 191), (183, 25), (123, 54), (85, 59), (72, 30),
          (167, 151)]  # replace with different list for tests
cityList = [City(i[0], i[1]) for i in cities]


bfr, brbg, bfbg, bdbg, params = geneticAlgorithm(
    100, numCities, 20, 1000, 0.005, cityList)
# print(bfr, "\n\n", brbg[0:5], "\n\n", bfbg[0:5], "\n\n", bdbg[0:5])


###Test for distancePlot###
distancePlot(bdbg, params)


###Test for evolutionPlot###
evolutionPlot(brbg, bdbg, cities)
