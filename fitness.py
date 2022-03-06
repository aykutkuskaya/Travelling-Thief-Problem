# İlk şehirden item alınmayacak
# Cities üzerinden crossover ve mutasyon yapılacak ama item da aynı şekilde değişecek
cities = [2, 3, 1]
items = [0, 1, 1]


# şehir sıralaması 1 2 3
coordinates = [[0, 1], [1, 2], [3, 3]]
item_value = [[3, 5], [2, 4], [8, 3]]

min_speed = 0.1
max_speed = 1

renting_ratio = 5.61
capacity_knapsack = 30

total_weight = 0
total_price = 0
time = 0
for index in range(len(cities)):
    if index == len(cities)-1:
        distance = ((coordinates[cities[index] - 1][0] - coordinates[cities[0]-1][0]) ** 2 + (
                    coordinates[cities[index] - 1][1] - coordinates[cities[0]-1][1]) ** 2) ** (1 / 2)
    else:
        distance = ((coordinates[cities[index] - 1][0]-coordinates[cities[index+1] - 1][0])**2+(coordinates[cities[index] - 1][1]-coordinates[cities[index+1] - 1][1])**2)**(1/2)

    if items[index] == 1:
        total_weight += item_value[cities[index]-1][1]
        total_price += item_value[cities[index]-1][0]

    speed = max_speed-(total_weight*(max_speed-min_speed)/capacity_knapsack)
    time += distance/speed

total_profit = total_price-(time*renting_ratio)

penalty = 0
if total_weight > capacity_knapsack:
    penalty = total_weight-capacity_knapsack

fitness = (1/total_profit) + penalty

print(total_profit)
print(penalty)
print(fitness)
