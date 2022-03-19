from __future__ import division
from ast import While
from cgi import print_form
from itertools import count
from pydoc import doc
import numpy
import random
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
                    self.wp.insert(i - 2, [p, w])   
def crossoverOrdered(ind1, ind2, items,items2):
    size = min(len(ind1), len(ind2))
    #buradaki random subsequence lenghtin kaç olacağı sorulacak (4)
    a=random.randint(0,size-4)
    print("Random index " + str(a))
    subsequence1=ind1[a:a+4]
    subsequence2=ind2[a:a+4]
    temp1=[0]*size
    temp2=[0]*size
    itemsNew=[0]*size
    itemsNew2=[0]*size
    temp1[a:a+4] = subsequence1
    temp2[a:a+4] = subsequence2
    itemsNew[a:a+4]=items[a:a+4]
    itemsNew2[a:a+4]=items2[a:a+4]
    checker=True
    counter=a+4
    adderIndex1=a+4
    while checker:
        if counter == size:       
            counter=0
      
        if ind2[counter] not in subsequence1 :
            temp1[adderIndex1] = ind2[counter]
            itemsNew[adderIndex1] =items[ind1.index(ind2[counter])]
            if adderIndex1 ==size-1:
                adderIndex1=0
            else:
                adderIndex1=adderIndex1+1
        if adderIndex1 ==a:
            checker=False
        counter=counter+1

    checker2=True
    counter2=a+4
    adderIndex2=a+4
    while checker2:
        if counter2 == size:
            counter2=0
        if ind1[counter2] not in subsequence2 :
            temp2[adderIndex2] = ind1[counter2]
            itemsNew2[adderIndex2] =items2[ind2.index(ind1[counter2])]
            if adderIndex2 ==size-1:
                adderIndex2=0
            else:
                adderIndex2=adderIndex2+1
        if adderIndex2 ==a:
            checker2=False
        counter2=counter2+1
    print("Subsequence1 : " + str(subsequence1))
    print("Subsequence2 : " + str(subsequence2))
    print("Result1 : " + str(temp1))
    print("Result2 : " + str(temp2))
    print("Items New  : " + str(itemsNew))
    print("Items New2  : " + str(itemsNew2))
    return ind1, ind2 ,itemsNew ,itemsNew2

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
def exchangeMutation(cities,items):
    geneNumbers = len(cities)
    x1, x2 = numpy.random.choice(range(geneNumbers), 2, False)
    print("Selected index :  ",x1)
    print("Selected index2 : ",x2)
    cities[x1], cities[x2] = cities[x2], cities[x1]
    items[x1], items[x2] =items[x2], items[x1]

    print(cities)
    print(items)

# If thief didn't collect any item, he picks an item randomly
def is_any_item(items):
    if 1 not in items:
        i = random.randint(1, len(items)-1)
        items[i] = 1
def calculateFitness(cities,items,item_value,distances):
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
        if items[index] == 1:
            total_weight += item_value[cities[index] - 2][1]
            total_price += item_value[cities[index] - 2][0]

        # Speed of thief is calculated according to formula.
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

    return fitness
    '''print(total_profit)
    print(penalty)
    print(fitness)'''
if __name__ == '__main__':

    # Read benchmark dataset and set variables
    file_name="st70_n69_uncorr_10.ttp"
    instance = Fitness(file_name)


    # İlk şehirden item alınmayacak
    # Cities üzerinden crossover ve mutasyon yapılacak ama item da aynı şekilde değişecek

    # Initialization of TSP solution (permutation representation)
    # !!!!İlk şehir her zaman 1. şehir
    tour = list(range(2, instance.dimension+1))
    random.shuffle(tour)
    tour2 = list(range(2, instance.dimension+1))
    random.shuffle(tour2)
    first_city = [1]
    cities1 = first_city+tour
    cities2 = first_city+tour2
    print("Cities 1 : "+str(cities1))
    print("Cities 2 : "+str(cities2))
    #Initialization of KP solution (binary representation)
    #!!!! İlk şehirden item alınmayacak
    initial_items = list(numpy.random.randint(2, size=instance.number_of_items))
    items = [0]+initial_items
    print("Items : " +str(items))

    initial_items2 = list(numpy.random.randint(2, size=instance.number_of_items))
    items2 = [0]+initial_items2
    print("Items2 : " +str(items2))

    
    # Check if thief collect an item or not
    is_any_item(items)

    ##crossoverOrdered(cities1,cities2,items,items2)
    ##exchangeMutation(cities1,items)
    # coordinates variable stores x and y values of each city. item_value variable stores weight and price of each item.
    # şehir sıralaması 1 2 3
    coordinates = instance.xy
    item_value = instance.wp

    # total_weight and total_price calculated according to collected items.
    total_weight = 0
    total_price = 0
    time = 0

    # calculates all distances
    distances = calculate_distances(coordinates)

    print("Calculated fitness for cities 1 : ",calculateFitness(cities1,items,item_value,distances))


