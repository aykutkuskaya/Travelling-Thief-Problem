from __future__ import division

import math
from ast import While
from cgi import print_form
from itertools import count
from pydoc import doc
import numpy
import random

global item_value
global coordinates
global distances

class Solution:
    def __init__(self,cities,items):
        self.cities =cities
        self.items = items
        self.fitness, self.profit = calculateFitness(self.cities, self.items, item_value, distances,itemsPerCity)


class Fitness:
 def __init__(self, file_name):
        self.tsp = False
        self.knapsack = False
        self.xy = []
        self.wp = []
        with open(file_name, 'r') as TTPInstance:
            for line in TTPInstance.readlines():
                line = line.replace('\n', '').replace('\t', ' ')
                if 'PROBLEM NAME:' in line:
                    self.problem_name = line.replace('PROBLEM NAME:', '').strip(' ')
                if 'KNAPSACK DATA TYPE:' in line:
                    self.knapsack_data_type = line.replace('KNAPSACK DATA TYPE:', '').strip(' ')
                if 'DIMENSION:' in line:
                    self.dimension = eval(line.replace('DIMENSION:', '').strip(' '))
                if 'NUMBER OF ITEMS:' in line:
                    self.number_of_items = eval(line.replace('NUMBER OF ITEMS:', '').strip(' '))
                if 'CAPACITY OF KNAPSACK:' in line:
                    self.capacity_of_knapsack = eval(line.replace('CAPACITY OF KNAPSACK:', '').strip(' '))
                if 'MIN SPEED:' in line:
                    self.min_speed = eval(line.replace('MIN SPEED:', '').strip(' '))
                if 'MAX SPEED:' in line:
                    self.max_speed = eval(line.replace('MAX SPEED:', '').strip(' '))
                if 'RENTING RATIO:' in line:
                    self.renting_ratio = eval(line.replace('RENTING RATIO:', '').strip(' '))
                if 'EDGE_WEIGHT_TYPE:' in line:
                    self.edge_weight_type = line.replace('EDGE_WEIGHT_TYPE:', '').strip(' ')
                if 'NODE_COORD_SECTION (INDEX, X, Y):' in line:
                    self.tsp = True
                    self.knapsack = False
                if 'ITEMS SECTION (INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):' in line:
                    self.tsp = False
                    self.knapsack = True
                if self.tsp and 'NODE_COORD_SECTION (INDEX, X, Y):' not in line:
                    i, x, y = eval(line.replace(' ', ','))
                    city = [x, y]
                    self.xy.append(city)
                if self.knapsack and 'ITEMS SECTION (INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):' not in line:
                    j, p, w, i = eval(line.replace(' ', ','))
                    self.wp.append([p, w, i])

# Calculates distance of each city and stores these values in 2-D array
def calculate_distances(coordinates):
    size = len(coordinates)
    distances = []

    for i in range(size):
        row = []
        for j in range(size):
            distance = ((coordinates[i][0] - coordinates[j][0]) ** 2 + (
                    coordinates[i][1] - coordinates[j][1]) ** 2) ** (1 / 2)
            row.append(distance)
        distances.append(row)

    return distances

def calculateFitness(cities,items,item_value,distances, itemsPerCity):
    # total_weight and total_price calculated according to collected items.
    total_weight = 0
    total_price = 0
    time = 0
    # This loop will be executed for each travel from one city to another one.
    for index in range(len(cities)):
        # If thief visited all cities he returns the initial city. Otherwise, he travels to next city.
        # This part calculates the total distance.
        if index == len(cities) - 1:
            distance = distances[cities[index] - 1][cities[0] - 1]
        else:
            distance = distances[cities[index] - 1][cities[index + 1] - 1]

        # If an item is collected total_weight and total_price variables are updated.
        for i in range(itemsPerCity):
            if items[index * itemsPerCity + i] == 1:
                total_weight += item_value[(cities[index] - 2) * itemsPerCity + i][1]
                total_price += item_value[(cities[index] - 2) * itemsPerCity + i][0]

        # Speed of thief is calculated according to formula.
        if total_weight > instance.capacity_of_knapsack:
            speed= instance.min_speed
        else:
            speed = instance.max_speed - (total_weight * (instance.max_speed - instance.min_speed)
                                      / instance.capacity_of_knapsack)
        # Spent time between two cities are added to total time.
        time += distance / speed

    # Profit of thief is calculated.
    total_profit = total_price - (time * instance.renting_ratio)

    # If thief exceed the capacity of knapsack a penalty value is added to result.
    penalty = 0
    if total_weight > instance.capacity_of_knapsack:
        penalty = total_weight - instance.capacity_of_knapsack

    # Calculation of fitness value
    fitness = (1 / total_profit) + penalty

    return fitness,total_profit
    '''print(total_profit)
    print(penalty)
    print(fitness)'''

def crossoverOrdered(ind1, ind2, items, items2,itemsPerCity):

    size = min(len(ind1), len(ind2))
    b=random.randint(1,int(size/2))
    a=random.randint(2,size-(b+1))
    subsequence1=ind1[a:a+b]
    subsequence2=ind2[a:a+b]
    temp1=[0]*size
    temp2=[0]*size
    itemsNew = [0] * size * itemsPerCity
    itemsNew2 = [0] * size * itemsPerCity
    temp1[a:a+b] = subsequence1
    temp2[a:a+b] = subsequence2
    itemsNew[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity]=items[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity]
    itemsNew2[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity]=items2[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity]

    checker=True
    counter=a+b
    adderIndex1=a+b
    while checker:
        if counter == size:
            temp1[0]=1
            itemsNew[0]=0
            counter = 1
        if counter == 0:
            temp1[0]=1
            itemsNew[0]=0
            counter=counter+1
        if adderIndex1 == 0:
            adderIndex1 =adderIndex1+1
        if ind2[counter] not in subsequence1 :
            temp1[adderIndex1] = ind2[counter]
            for i in range(itemsPerCity):
                itemsNew[adderIndex1*itemsPerCity+i] =items2[ind2.index(ind2[counter])*itemsPerCity+i]
            if adderIndex1 ==size-1:
                adderIndex1=0
            else:
                adderIndex1=adderIndex1+1
        if adderIndex1 ==a:
            checker=False
        counter=counter+1

    checker2=True
    counter2=a+b
    adderIndex2=a+b
    while checker2:
        if counter2 == size:
            temp2[0]=1
            itemsNew2[0]=0
            counter2=1
        if counter2 == 0:
            temp2[0]=1
            itemsNew2[0]=0
            counter2=counter2+1
        if adderIndex2 == 0:
            adderIndex2 =adderIndex2+1
        if ind1[counter2] not in subsequence2 :
            temp2[adderIndex2] = ind1[counter2]
            for i in range(itemsPerCity):
                itemsNew2[adderIndex2 * itemsPerCity + i] = items[ind1.index(ind1[counter2]) * itemsPerCity + i]
            if adderIndex2 ==size-1:
                adderIndex2=0
            else:
                adderIndex2=adderIndex2+1
        if adderIndex2 ==a:
            checker2=False
        counter2=counter2+1

    return temp1, temp2, itemsNew, itemsNew2


def crossoverPartialMapped(ind1, ind2, items, items2,itemsPerCity):
    size = min(len(ind1), len(ind2))
    b = random.randint(1, int(size / 2))
    a = random.randint(2, size - (b + 1))
    subsequence1 = ind1[a:a + b]
    subsequence2 = ind2[a:a + b]
    temp1 = [0] * size
    temp2 = [0] * size
    itemsNew = [0]*size*itemsPerCity
    itemsNew2 = [0]*size*itemsPerCity
    temp1[a:a+b] = subsequence2
    temp2[a:a+b] = subsequence1
    itemsNew[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity] = items2[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity]
    itemsNew2[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity] = items[a*itemsPerCity:a*itemsPerCity+b*itemsPerCity]
    checker=True
    counter=a+b
    adderIndex1=a+b
    while checker:
        if counter == size:
            counter=0
        temporary=ind1[counter]
        if temporary  in subsequence2 :
            while(temporary  in subsequence2):
                temporary=subsequence1[subsequence2.index(temporary)]
        temp1[adderIndex1] = temporary
        for i in range(itemsPerCity):
            itemsNew[adderIndex1*itemsPerCity+i] =items[ind1.index(temporary)*itemsPerCity+i]
        if adderIndex1 ==size-1:
                adderIndex1=0
        else:
            adderIndex1=adderIndex1+1
        if adderIndex1 ==a:
            checker=False
        counter=counter+1

    checker2=True
    counter2=a+b
    adderIndex2=a+b
    while checker2:
        if counter2 == size:
            counter2=0
        temporary=ind2[counter2]
        if temporary  in subsequence1 :
            while(temporary  in subsequence1):
                temporary=subsequence2[subsequence1.index(temporary)]
        temp2[adderIndex2] = temporary
        for i in range(itemsPerCity):
            itemsNew2[adderIndex2*itemsPerCity+i] =items2[ind2.index(temporary)*itemsPerCity+i]
        if adderIndex2 ==size-1:
                adderIndex2=0
        else:
            adderIndex2=adderIndex2+1
        if adderIndex2 ==a:
            checker2=False
        counter2=counter2+1
    return temp1,temp2,itemsNew,itemsNew2



def exchangeMutation(cities,items, itemsPerCity):
    geneNumbers = len(cities)
    itemsNew=[0]*geneNumbers*itemsPerCity
    citiesNew=[0]*geneNumbers
    itemsNew[0:geneNumbers*itemsPerCity]=items[0:geneNumbers*itemsPerCity]
    citiesNew[0:geneNumbers]=cities[0:geneNumbers]
    x1, x2 = numpy.random.choice(range(1,geneNumbers), 2, replace=False)
    citiesNew[x1], citiesNew[x2] = cities[x2], cities[x1]
    for i in range(itemsPerCity):
        itemsNew[x1 * itemsPerCity + i], itemsNew[x2 * itemsPerCity + i] = items[x2 * itemsPerCity + i], items[x1 * itemsPerCity + i]
    return citiesNew, itemsNew


def inversionMutation(cities, items, itemsPerCity):
    array = cities
    array1 = items
    l = len(array)
    r1, r2 = numpy.random.choice(range(1, l-1), 2, replace=False)
    while (r1 >= r2):
        r1, r2 = numpy.random.choice(range(1, l-1), 2, replace=False)
    mid = r1 + ((r2 + 1) - r1) / 2
    endCount = r2
    for i in range(r1, int(mid)):
        tmp = array[i]
        array[i] = array[endCount]
        array[endCount] = tmp
        for j in range(itemsPerCity):
            tmp = array1[i * itemsPerCity + j]
            array1[i * itemsPerCity + j] = array1[endCount * itemsPerCity + j]
            array1[endCount * itemsPerCity + j] = tmp
        endCount = endCount - 1
    return(array, array1)


# If thief didn't collect any item, he picks an item randomly
def is_any_item(items):
    if 1 not in items:
        i = random.randint(1, len(items)-1)
        items[i] = 1



def updatePopulation(solutions,k):
    '''solutions = sorted(solutions, key=lambda solutions: solutions.fitness, reverse=True)
    while len(solutions)!=100:
        solutions.pop()
    return solutions'''
    solutions = sorted(solutions, key=lambda solutions: solutions.fitness)
    positives = list(filter(lambda x: (x.fitness>= 0), solutions))
    negatives = list(filter(lambda x: (x.fitness < 0), solutions))
    solutions= positives+negatives
    while len(solutions) != k:
        solutions.pop()
    return solutions


def selectParents(solutions):
    group1 = []
    group2 = []
    r = numpy.random.choice(range(0, len(solutions)), 8, replace=False)
    for i in range(0, 4):
        group1.append(solutions[r[i]])
    for i in range(4, 8):
        group2.append(solutions[r[i]])

    group1 = sorted(group1, key=lambda group1: group1.fitness)
    positives = list(filter(lambda x: (x.fitness >= 0), group1))
    negatives = list(filter(lambda x: (x.fitness < 0), group1))
    group1 = positives + negatives

    group2 = sorted(group2, key=lambda group2: group2.fitness)
    positives = list(filter(lambda x: (x.fitness >= 0), group2))
    negatives = list(filter(lambda x: (x.fitness <0), group2))
    group2 = positives + negatives
    parent1 = Solution(group1[0].cities.copy(),group1[0].items.copy())
    parent2 = Solution(group2[0].cities.copy(),group2[0].items.copy())

    '''while parent1 == parent2:
        group1 = []
        group2 = []
        for i in range(0, 4):
            group1.append(solutions[random.randint(0, len(solutions) - 1)])
            group2.append(solutions[random.randint(0, len(solutions) - 1)])
        group1 = sorted(group1, key=lambda group1: group1.fitness, reverse=True)
        group2 = sorted(group2, key=lambda group2: group2.fitness, reverse=True)
        parent1 = group1[0]
        parent2 = group2[0]'''
    return parent1, parent2


if __name__ == '__main__':
    # Read benchmark dataset and set variables
    file_name = "st70_n207_bounded-strongly-corr_01.ttp"
    instance = Fitness(file_name)

    coordinates = instance.xy
    item_value = instance.wp
    distances = calculate_distances(coordinates)
    itemsPerCity = math.ceil(instance.number_of_items/instance.dimension)
    a = instance.wp
    instance.wp = sorted(a, key=lambda a: a[2])

    k = 100
    probabiltyCrossover = 90
    probabiltyMutation = 10
    solutions = []
    for i in range(k):
        tour = list(range(2, instance.dimension + 1))
        random.shuffle(tour)
        first_city = [1]
        cities = first_city + tour

        # Initialization of KP solution (binary representation)

        initial_items = list(numpy.random.randint(2, size=instance.number_of_items))
        items = [0]*itemsPerCity + initial_items
        # Check if thief collect an item or not
        is_any_item(items)
        solutions.append(Solution(cities, items))




    for counter in range(200000):

        isCrossover=False
        isMutation=False
        parent1, parent2 = selectParents(solutions)
        cities1, cities2, items1, items2 = parent1.cities.copy(), parent2.cities.copy(), parent1.items.copy(), parent2.items.copy()
        if random.randint(1, 101) < probabiltyCrossover:
            cities1, cities2, items1, items2 = crossoverPartialMapped(parent1.cities.copy(), parent2.cities.copy(), parent1.items.copy(), parent2.items.copy(), itemsPerCity)
            isCrossover=True

        if random.randint(1, 101) < probabiltyMutation:
            cities1, items1 = inversionMutation(cities1.copy(), items1.copy(),itemsPerCity)
            cities2, items2 = inversionMutation(cities2.copy(), items2.copy(),itemsPerCity)
            isMutation=True
        child1 = Solution(cities1.copy(),items1.copy())
        child2 = Solution(cities2.copy(),items2.copy())

        if isCrossover or isMutation:
            different=0
            for i in solutions:
                if i.fitness!=child1.fitness:
                    different+=1
            if different==len(solutions):
                solutions.append(child1)
            different = 0
            for i in solutions:
                if i.fitness != child2.fitness:
                    different += 1
            if different ==len(solutions):
                solutions.append(child2)


            solutions=updatePopulation(solutions,k)
        for i in solutions:
            print(i.fitness,i.profit,i.cities)
