# CS461-GeneticAlgorithm

This program was made for an AI class to implement a genetic algorithm for scheduling using python.

The program runs at minimum 100 generations or until the fitness score has stabilized improving. At the end, the program should produce an "output.txt" file in the same directory it was run.

Sample output:
![OutputSample](https://github.com/user-attachments/assets/d84fc041-6f51-465a-baf8-61b172bc5e7f)

Current issues:
A part of the fitness function, in the other_adjustment_fitness, is potentially not implemented correctly, as SLA191x can be scheduled right after SLA100x in the output
