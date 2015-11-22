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
        return population

    # def fitness_funct():
    #     return fitness

    def cluster(self):
        pop = self.select_pop()
        pBest = pop
        pBestFitness = [1000] * len(pop)
        gBest = []
        gBestFitness = [1000]
        for particle in pop:
            clusters = []
            for i in range(self.clusters):
                c = []
                clusters.append(c)
            for d in self.dataset:
                closest = [1000, 0]
                for i in range(len(particle)):
                    dist = np.linalg.norm(d.features - particle[i].features)
                    if dist < closest[0]:
                        closest = [dist, i]
                clusters[closest[1]].append(d)
            for i in range(len(clusters)):
                print i, len(clusters[i])


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

