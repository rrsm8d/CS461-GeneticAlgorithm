import random
import numpy as np
from scipy.special import softmax
import math
import numpy as np

# Constants
# Some activities have two sections, A and B
activities = [ 
    "SLA100A",
    "SLA100B",
    "SLA191A",
    "SLA191B",
    "SLA201",
    "SLA291",
    "SLA303",
    "SLA304",
    "SLA394",
    "SLA449",
    "SLA451"
]

rooms = [
    "Slater 003",
    "Roman 216",
    "Loft 206",
    "Roman 201",
    "Loft 310",
    "Beach 201",
    "Beach 301",
    "Logos 325",
    "Frank 119"
]

# room : capacity
roomsCapacity = {
    "Slater 003": 45,
    "Roman 216": 30,
    "Loft 206": 75,
    "Roman 201": 50,
    "Loft 310": 108,
    "Beach 201": 60,
    "Beach 301": 75,
    "Logos 325": 450,
    "Frank 119": 60
}

# Military time
times = [
    10,
    11,
    12,
    13,
    14,
    15
]

facilitators = [
    "Lock",
    "Glen",
    "Banks",
    "Richards",
    "Shaw",
    "Singer",
    "Uther",
    "Tyler",
    "Numen",
    "Zeldin"
]

# Use preferred and other listing for scoring purposes
# 

extendedActivityInfo = {
    "SLA100A": {
        "Expected": 50,
        "Preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "Other": ["Numen", "Richards"]
    },
    "SLA100B": {
        "Expected": 50,
        "Preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "Other": ["Numen", "Richards"]
    },
    "SLA191A": {
        "Expected": 50,
        "Preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "Other": ["Numen", "Richards"]
    },
    "SLA191B": {
        "Expected": 50,
        "Preferred": ["Glen", "Lock", "Banks", "Zeldin"],
        "Other": ["Numen", "Richards"]
    },
    "SLA201": {
        "Expected": 50,
        "Preferred": ["Glen", "Banks", "Zeldin", "Shaw"],
        "Other": ["Numen", "Richards", "Singer"]
    },
    "SLA291": {
        "Expected": 50,
        "Preferred": ["Lock", "Banks", "Zeldin", "Singer"],
        "Other": ["Numen", "Richards", "Shaw", "Tyler"]
    },
    "SLA303": {
        "Expected": 60,
        "Preferred": ["Glen", "Zeldin", "Banks"],
        "Other": ["Numen", "Singer", "Shaw"]
    },
    "SLA304": {
        "Expected": 25,
        "Preferred": ["Glen", "Banks", "Tyler"],
        "Other": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]
    },
    "SLA394": {
        "Expected": 20,
        "Preferred": ["Tyler", "Singer"],
        "Other": ["Richards", "Zeldin"]
    },
    "SLA449": {
        "Expected": 60,
        "Preferred": ["Tyler", "Singer", "Shaw"],
        "Other": ["Zeldin", "Uther"]
    },
    "SLA451": {
        "Expected": 100,
        "Preferred": ["Tyler", "Singer", "Shaw"],
        "Other": ["Zeldin", "Uther", "Richards", "Banks"]
    }
}

# Need to assign for each activity, A room, time, facilitator
# Initially random, with random population >= 500. Apply genetic algos to improve it.
# Use softmax normalization

# PRE: None
# POST: Returns a list of lists, containing the schedule assigned to every activity
def randomize_schedule():
    schedules = []
    # Randomly select the time, facilitator, and room for each activity
    for activity in activities:
        time = random.choice(times)
        room = random.choice(rooms)
        # Does not need to be from Preferred or Expected
        # Fitness function will assess and score
        facilitator = random.choice(facilitators)
        schedule = [activity, room, time, facilitator]
        schedules.append(schedule)
    return schedules

# PRE: A list containing the schedule, the full activity info, and the list of each rooms capacity
# POST: Return score
def room_size_fitness(schedule, extendedActivityInfo, roomsCapacity):
    # schedule = [ activity, room, time, facilitator]
    activity_name = schedule[0]
    expected_enrollment = extendedActivityInfo[activity_name]['Expected'] # The expected enrollment for an activity
    room_name = schedule[1]
    capacity = roomsCapacity[room_name] # the max that a room can hold

    if expected_enrollment * 6 <= capacity: # Too big of a room
        return -0.4
    elif expected_enrollment * 3 <= capacity: # Too big of a room
        return -0.2
    elif expected_enrollment > capacity: # Too small of a room
        return -0.5
    else: # Perfect
        return 0.3 

# PRE: a schedule, the index, and the list of schedules
# POST: return the fitness score based on overlaps
def overlap_fitness(schedule, index, schedules):
    totalOverlapFitness = 0.0
    for i in range(index):
        # other_activity = [ activity, room, time, facilitator]
        other_activity = schedules[i]
        # Room and time slot
        # Since activities are 50 minutes, only need to compare 1 number
        if ( schedule[1] == other_activity[1] and schedule[2] == other_activity[2] ): # This time slot is being used
            totalOverlapFitness -= 0.5
        else:
            totalOverlapFitness += 0.0 # No overlap
    return totalOverlapFitness

def load_fitness(schedule, schedules):
    # Track the facilitators
    facilitator_load = {}
    for activity in schedules:
        if activity[2] == schedule[2]: # Time slot
            if activity[3] not in facilitator_load:
                facilitator_load[activity[3]] = 1
            else:
                facilitator_load[activity[3]] += 1

    totalLoadFitness = 0.0

    # For only 1 activity in the time slot [good]
    if (schedule[3] in facilitator_load and facilitator_load[schedule[3]] == 1):
        totalLoadFitness += 0.2
    # For more than 1 activity in the time slot [bad]
    if (schedule[3] in facilitator_load and facilitator_load[schedule[3]] > 1):
        totalLoadFitness -= 0.2
    # A facilitator has more than 4 activities
    if ( (sum(facilitator_load.values())) > 4):
        totalLoadFitness -= 0.5
    # A single facilitator has 1 or 2 activities
    if ( (sum(facilitator_load.values())) <= 2):
        if (not schedule[3] == 'Tyler'): # Dr. Tyler exception
            totalLoadFitness -= 0.4

    return totalLoadFitness

def has_consecutive_activites(schedule, schedules):
    for activity in schedules:
        if(activity[3] == schedule[3] and abs(activity[2] - schedule[2]) == 1):
            return True
    return False

def consecutive_fitness(schedule, schedules):
    roomList = ["Roman 201", "Beach 201","Roman 316", "Beach 301"]
    totalConsecutiveFitness = 0.0
    for activity in schedules:
        if(has_consecutive_activites(schedule, schedules)):
            # one of the activities is in Roman or Beach, and the other isn’t
            if (activity[1] in roomList and schedule[1] not in roomList) or (activity[1] not in roomList and schedule[1] in roomList):
                totalConsecutiveFitness -= 0.4
            else:
                totalConsecutiveFitness += 0.5
    return totalConsecutiveFitness

def other_adjustment_fitness(schedule, schedules):
    totalAdjustmentFitness = 0.0

    if schedule[0] == "SLA100A" or schedule[0] == "SLA100B":
        other = [act for act in schedules if act[0] == "SLA100A" or "SLA100B" and act != schedule]

        for other_activity in other:
            time_difference = abs(schedule[2] - other_activity[2])
            if time_difference > 4: # 4 hour difference
                totalAdjustmentFitness += 0.5
            if schedule[2] == other_activity[2]: # Same slot
                totalAdjustmentFitness -= 0.5

    elif schedule[0] == "SLA191A" or schedule[0] == "SLA191B":
        other = [act for act in schedules if act[0] == "SLA191A" or "SLA191B" and act != schedule]

        for other_activity in other:
            time_difference = abs(schedule[2] - other_activity[2])
            if time_difference > 4: # 4 hour difference
                totalAdjustmentFitness += 0.5
            if schedule[2] == other_activity[2]: # Same slot
                totalAdjustmentFitness -= 0.5
    
    return totalAdjustmentFitness


# PRE: A list of lists, containing the schedules
# POST: Return a number of the fitness score
def fitness_assessment(schedules):
    total_fitness = 0
    for i, schedule in enumerate(schedules):
        # schedule = [ activity, room, time, facilitator]
        room_fitness = room_size_fitness(schedule, extendedActivityInfo, roomsCapacity)
        total_fitness += room_fitness
        total_fitness += overlap_fitness(schedule, i, schedules)
        
        #Check facilitator conditions
        facilitator = schedule[3]
        
        for x in extendedActivityInfo[schedule[0]]["Preferred"]: #extendedActivityInfo[activity]["Preferred"]
            for y in extendedActivityInfo[schedule[0]]["Other"]: #extendedActivityInfo[activity]["Other"]
                if facilitator == x: # Preferred
                    total_fitness += 0.5
                elif facilitator == y: # Other
                    total_fitness += 0.2
                else: # Bad facilitator
                    total_fitness -= 0.1
        
        
        total_fitness += load_fitness(schedule,schedules)
        total_fitness += consecutive_fitness(schedule, schedules)
        total_fitness += other_adjustment_fitness(schedule, schedules)
        
    return total_fitness


############################################################################################################
# Genetic Algorithm

def selection(initialPopulation, softmaxFitness, amount):
    selectedIndices = np.random.choice(len(initialPopulation), size=amount, p=softmaxFitness)
    # Retrieve the selected individuals
    selectedIndividuals = [initialPopulation[i] for i in selectedIndices]
    
    return selectedIndividuals

def crossover(p1, p2):
    # Made with help from GPT-4o-latest
    crossover_point = random.randint(1, len(p1))  # Choose crossover point
    offspring1 = p1[:crossover_point] + p2[crossover_point:]
    offspring2 = p2[:crossover_point] + p1[crossover_point:]
    return offspring1, offspring2

def mutate(mutationRate, child):
    newSchedule = child.copy()
    for i in range(len(child)):
        if random.random() < mutationRate:
            # random selection
            newActivity = random.choice(activities)
            newRoom = random.choice(rooms)
            newTime = random.choice(times)
            newFacilitator = random.choice(facilitators)
            newSchedule[i] = (newSchedule[i][0], newRoom, newTime, newFacilitator)
    return newSchedule


def parentSelection(selectedPopulation, selectedSoftmaxFitness):
    # Made with help from GPT-4o-latest
    selected = random.uniform(0, 1)
    cumulative_probability = 0

    for i in range(len(selectedPopulation)):
        cumulative_probability += selectedSoftmaxFitness[i]
        if cumulative_probability > selected:
            return selectedPopulation[i]

def crossoverAndMutate(selectedPopulation, selectedSoftmaxFitness, mutationRate, crossoverRate):
    mutated = []
    # Steps of 2, will just crossover the two adjacent parents in index
    for i in range(0, len(selectedPopulation), 2):
        if (i+1) < len(selectedPopulation): # Don't want to go OOB
            p1 = parentSelection(selectedPopulation, selectedSoftmaxFitness)
            p2 = parentSelection(selectedPopulation, selectedSoftmaxFitness)
            # Should be 0.5 unless I change it
            if random.random() < crossoverRate:
                c1, c2 = crossover(p1, p2)
            else:
                c1 = p1
                c2 = p2
            # Children selected

            # Don't want the population for next gen to shrink
            m1 = mutate(mutationRate, c1)
            m2 = mutate(mutationRate, c2)
            # Doesn't work properly for future generations
            # mutated.extend(m1)
            # mutated.extend(m2)

            # ***DO NOT CHANGE THIS***
            mutated.extend([m1, m2]) # Even though I never formatted my original list like this ??????
    return mutated



############################################################################################################
# Main program

# Schedules should be a list of lists
initialPopulation = [randomize_schedule() for i in range(500)]
mutationRate = 0.01
crossoverRate = 0.5
avgFitnessHistory = []
index = 0
# 100 generations
for gen in range(500):
    avgFitness = 0
    # Calculate the fitness
    fitnessScores = [fitness_assessment(schedule) for schedule in initialPopulation]
    # I kept ending up with unusual floating point precision errors, and can't find a pretty solution, so the scores are just rounded to 2 decimal places
    roundedFitnessScores = [round(score, 2) for score in fitnessScores]

    avgFitness = np.mean(roundedFitnessScores)
    avgFitnessHistory.append(avgFitness)

    softmaxFitness = softmax(roundedFitnessScores)
    selectedPopulation = selection(initialPopulation, softmaxFitness, 250) 
    
    selectedFitnessScores = [fitness_assessment(schedule) for schedule in selectedPopulation]
    roundedSelectedFitnessScores = [round(score, 2) for score in selectedFitnessScores]
    selectedSoftmaxFitness = softmax(roundedSelectedFitnessScores)

    offspring = crossoverAndMutate(selectedPopulation, selectedSoftmaxFitness, mutationRate, crossoverRate) # crossover rate as well

    initialPopulation = offspring
    # Optional: Track progress or log statistics here
    if gen > 1:
        if (avgFitnessHistory[index] > avgFitnessHistory[index-1]):
            mutationRate /= 2
        if (avgFitnessHistory[index] - avgFitnessHistory[index-1] <= 0.01) and gen >= 100: # Continue until the improvement in average fitness for generation G is less than 1%
            print("The improvement has stagnated after 100 generations")
            break

    print(f"Generation {gen + 1} complete.")
    index += 1

#####################################################################################
# After completion

finalFitnessScores = []
maximum = 0
maxIndex = 0
for i, schedule in enumerate(initialPopulation):
    fit = fitness_assessment(schedule)
    if fit > maximum:
        maximum = fit
        maxIndex = i

print("Writing to file...")

# Open a file in write mode
with open("output.txt", "w") as file:
    for row in initialPopulation[maxIndex]:
        # Convert each row to a string with elements separated by spaces and write it to the file
        file.write(" ".join(map(str, row)) + "\n")


print("Written to output.txt")