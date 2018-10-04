from GA import *


# Test for City class

# Houston = City(1, 2.5)
# Boston = City(20.2, 12)
# print(Houston)
# print(type(Houston))
# print(Houston.distance(Boston))

# Generic list of cities

# cityList = []
# for i in range(8):
#     cityList.append(City(x=round(random.random()*200), y=round(random.random()*200)))
# R1 = Route(cityList)
# print(R1.routeFitness())
# print(R1.distance)
# print(cityList)

# Test for initialPopulation

# abc = initialPopulation(5, 20)
# print(abc,"\n")

# Test for rankRoutes

# abc = initialPopulation(5, 20)
# print(abc,"\n")
# cba = rankRoutes(abc)
# print(cba)
# print("\n")

# Test for selection

# abc = initialPopulation(5, 20)
# print(abc,"\n")
# cba = rankRoutes(abc)
# print(cba)
# numElites = 2
# j = selection(cba,numElites)
# print(j)
# print("\n")

# Test for matingPool

# abc = initialPopulation(6, 20)
# print(abc, "\n")
# cba = rankRoutes(abc)
# print(cba)
# numElites = 2
# j = selection(cba,numElites)
# print(j)
# pool = matingPool(abc, j)
# print(pool)
# print("\n")

# Test for breed

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

# Test for breedPopulation

abc = initialPopulation(6, 10)
#print(abc, "\n")
cba = rankRoutes(abc)
# print(cba)
numElites = 2
j = selection(cba, numElites)
# print(j)
pool = matingPool(abc, j)
# print(pool)
next_gen = breedPopulation(pool, numElites)
for individual in next_gen:
    print(individual, ' ')
# note: something feels a little inbred
