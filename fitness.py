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
    a=random.randint(0,size-5)
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
    return ind1, ind2 ,itemsNew ,itemsNew2
def crossoverPartialMapped(ind1, ind2, items,items2):
    size = min(len(ind1), len(ind2))
    a=random.randint(0,size-5)
    subsequence1=ind1[a:a+4]
    subsequence2=ind2[a:a+4]
    temp1=[0]*size
    temp2=[0]*size
    itemsNew=[0]*size
    itemsNew2=[0]*size
    temp1[a:a+4] = subsequence2
    temp2[a:a+4] = subsequence1
    itemsNew[a:a+4]=items2[a:a+4]
    itemsNew2[a:a+4]=items[a:a+4]
    checker=True
    counter=a+4
    adderIndex1=a+4
    while checker:
        if counter == size:       
            counter=0
        temporary=ind1[counter]
        if temporary  in subsequence2 :
            while(temporary  in subsequence2):
                temporary=subsequence1[subsequence2.index(temporary)]
        temp1[adderIndex1] = temporary
       #düzeltilecek ----- itemsNew[adderIndex1] =items[ind1.index(ind2[counter])]
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
        temporary=ind2[counter2]
        if temporary  in subsequence1 :
            while(temporary  in subsequence1):
                temporary=subsequence2[subsequence1.index(temporary)]
        temp2[adderIndex2] = temporary
       #düzeltilecek ----- itemsNew[adderIndex1] =items[ind1.index(ind2[counter])]
        if adderIndex2 ==size-1:
                adderIndex2=0
        else:
            adderIndex2=adderIndex2+1
        if adderIndex2 ==a:
            checker2=False
        counter2=counter2+1
    return temp1,temp2
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
    itemsNew=[0]*geneNumbers
    citiesNew=[0]*geneNumbers
    itemsNew[0:geneNumbers]=items[0:geneNumbers]   
    citiesNew[0:geneNumbers]=cities[0:geneNumbers]
    x1, x2 = numpy.random.choice(range(geneNumbers), 2, False)
    citiesNew[x1], citiesNew[x2] = cities[x2], cities[x1]
    itemsNew[x1], itemsNew[x2] =items[x2], items[x1]
    return citiesNew, itemsNew

def inversionMutation(cities):
    array = cities
    l = len(array)
    r1 = random.randint(0, l)
    r2 = random.randint(0,l)
    while (r1 >= r2):
        r1 = random.randint(0, l)
        r2 = random.randint(0, l)
    mid = r1 + ((r2 + 1) - r1) / 2
    endCount = r2
    for i in range(r1, int(mid)):
        tmp = array[i]
        array[i] = array[endCount]
        array[endCount] = tmp
        endCount = endCount - 1
    return(array)
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

    return total_profit
    '''print(total_profit)
    print(penalty)
    print(fitness)'''
if __name__ == '__main__':
    # Read benchmark dataset and set variables
    file_name = "st70_n69_uncorr_10.ttp"
    instance = Fitness(file_name)

    # İlk şehirden item alınmayacak
    # Cities üzerinden crossover ve mutasyon yapılacak ama item da aynı şekilde değişecek

    # Initialization of TSP solution (permutation representation)
    # !!!!İlk şehir her zaman 1. şehir
    tour = list(range(2, instance.dimension + 1))
    random.shuffle(tour)
    tour2 = list(range(2, instance.dimension + 1))
    random.shuffle(tour2)
    first_city = [1]
    cities1 = first_city + tour
    cities2 = first_city + tour2




    # Initialization of KP solution (binary representation)
    # !!!! İlk şehirden item alınmayacak
    initial_items = list(numpy.random.randint(2, size=instance.number_of_items))
    items1 = [0] + initial_items

    initial_items2 = list(numpy.random.randint(2, size=instance.number_of_items))
    items2 = [0] + initial_items2




    # Check if thief collect an item or not
    is_any_item(items1)
    is_any_item(items2)

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
    print("Cities 1 : ", cities1)
    print("Items : ", items1)
    profit1=calculateFitness(cities1, items1, item_value, distances)
    print("Calculated fitness for cities 1 : ", profit1)
    print("Cities 2 : ", cities2)
    print("Items 2: ", items2)
    profit2= calculateFitness(cities2, items2, item_value, distances)
    print("Calculated fitness for cities 2 : ", profit2 )


    for i in range(100):
        print("----------------------------------------------------------------------------------------------")
        cities_child1, cities_child2, items_child1, items_child2 = crossoverOrdered(cities1, cities2, items1, items2)
        profit_child1 = calculateFitness(cities_child1, items_child1, item_value, distances)
        profit_child2 = calculateFitness(cities_child2, items_child2, item_value, distances)
        print("Calculated fitness for cities child 1 : ",profit_child1)
        print("Calculated fitness for cities child 2 : ",profit_child2)
    
        cities_child1,items_child1 = exchangeMutation(cities_child1,items_child1)
        profit_child1=calculateFitness(cities_child1, items_child1, item_value, distances)
        print("Calculated fitness for cities child 1 : ", profit_child1)
        cities_child2, items_child2 = exchangeMutation(cities_child2, items_child2)
        profit_child2 = calculateFitness(cities_child2, items_child2, item_value, distances)
        print("Calculated fitness for cities child 2 : ", profit_child2)

        if profit_child1>profit1:
            cities1=cities_child1
            items1=items_child1
            profit1=profit_child1
            print("***Child 1 is new parent***")
        if profit_child2>profit2:
            cities2=cities_child2
            items2 = items_child2
            profit2=profit_child2
            print("***Child 2 is new parent***")

    print(calculateFitness(cities1,items1,item_value,distances))
    print(calculateFitness(cities2, items2,item_value ,distances))


