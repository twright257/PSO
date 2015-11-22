__author__ = 'Toasty'
import numpy as np
import random

class ParticleSwarm:

    def __init__(self, dataset, popSize, clusters):
        self.dataset = dataset
        self.popSize = popSize
        self.clusters = clusters
        self.cluster()

    def select_pop(self):
        population = []
        for p in range(self.popSize):
            particle = []
            for c in range(self.clusters):
                randIndex = random.randint(0, len(self.dataset) - 1)
                particle.append(self.dataset[randIndex])
            population.append(particle)
        for p in population:
            print p

        return population

    # def fitness_funct():
    #     return fitness

    def cluster(self):
        pBest = self.select_pop()
        gBest = []




class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in out training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets = np.array(target)

def main():
    #create dataset
    dataset = []
    with open('.\Data\NewSpect.txt', 'r') as dataIn:
        for line in dataIn:
            variables = line.split(" ")
            classifier = []
            for elem in variables[-1].split(","):
                classifier.append(elem)
            classifier[0] = classifier[0][1]
            classifier[-1] = classifier[-1][0]
            variables.pop()
            classifier = map(float, classifier)
            variables = map(float, variables)
            data = Instance(variables, classifier)
            dataset.append(data)

    pso = ParticleSwarm(dataset, 10, 3)


if __name__ == '__main__':
    main()

