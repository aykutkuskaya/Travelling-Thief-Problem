from __future__ import division

import math
from ast import While
from cgi import print_form
from itertools import count
from operator import index
from pydoc import doc
from unittest import result
import numpy
import random

global item_value
global coordinates
global distances

class Solution:
    def __init__(self,cities,items):
        self.cities =cities
        self.items = items
        self.trialCounter=0
        self.probability=0
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

    if fitness >= 0:
        return 1/(1 + fitness), total_profit
    else:
        return 1 + abs(fitness), total_profit


def inversionMutation(cities, items, itemsPerCity):
    array = cities.copy()
    array1 = items.copy()
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


def calculateAllFitness(solutions):
    result=0
    for s in solutions:
        result=result+s.fitness
    return result
def calculateProbability(solutions):
    allFitness=calculateAllFitness(solutions)
    temp=0
    for s in solutions:
        ratioFitness=s.fitness / allFitness
        s.probability =temp + ratioFitness
        temp = s.probability
def randomSelect(solutions):
    choice=random.uniform(0,1)
    index=0
    for i in solutions:
        if(choice <=i.probability):
            return i,index
        index=index+1

def employeeBee(solutions):
    for s in range(len(solutions)):
        mutationed=inversionMutation(solutions[s].cities,solutions[s].items,itemsPerCity)
        result =Solution(mutationed[0],mutationed[1])
        if result.fitness > solutions[s].fitness:
            solutions[s]=result
        else:
            solutions[s].trialCounter=solutions[s].trialCounter+1
        
    return solutions        
def onLookerBee(solutions):
    calculateProbability(solutions)
    for i in range(len(solutions)):
        selectedSolution,index=randomSelect(solutions)
        mutationed=inversionMutation(selectedSolution.cities,selectedSolution.items,itemsPerCity)
        result =Solution(mutationed[0],mutationed[1])

        if result.fitness > selectedSolution.fitness:
            solutions[index]=result
        else:
            solutions[index].trialCounter=solutions[index].trialCounter+1
        calculateProbability(solutions)
    return solutions
def scoutBee(solutions,limit,bestSolution):
    for i in range(len(solutions)):
        if solutions[i].trialCounter >= limit:
            tour = list(range(2, instance.dimension + 1))
            random.shuffle(tour)
            first_city = [1]
            cities = first_city + tour

            # Initialization of KP solution (binary representation)

            initial_items = list(numpy.random.randint(2, size=instance.number_of_items))
            items = [0]*itemsPerCity + initial_items
            # Check if thief collect an item or not
            is_any_item(items)
            solutions[i] = Solution(cities,items)

    calculateAllFitness(solutions)
    calculateProbability(solutions)     
    fitnessesList= [x.fitness for x in solutions]
    
    tempbest=solutions[fitnessesList.index(max(fitnessesList))]
    if(bestSolution.fitness < tempbest.fitness):
        bestSolution= tempbest
    return solutions,bestSolution
if __name__ == '__main__':
    # Read benchmark dataset and set variables
    file_name = "st70_n69_uncorr_10.ttp"
    instance = Fitness(file_name)

    coordinates = instance.xy
    item_value = instance.wp
    distances = calculate_distances(coordinates)
    itemsPerCity = math.ceil(instance.number_of_items/instance.dimension)
    a = instance.wp
    instance.wp = sorted(a, key=lambda a: a[2])

    k = 100
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
    fitnessesList= [x.fitness for x in solutions]
    bestSolution=solutions[fitnessesList.index(max(fitnessesList))]
    limit=100
    iterationCount=10000
    for i in range(iterationCount):
        employeeBee(solutions)   
        onLookerBee(solutions)
        result=scoutBee(solutions,limit,bestSolution)
        bestSolution=result[1]
        print("Best Solution , ",bestSolution.fitness , "Total profit ," , bestSolution.profit)
        print("--------------------------------------------")
    for s in solutions:
        print("Trial Counter",s.trialCounter)
        print("Fitness ",s.fitness)
  

