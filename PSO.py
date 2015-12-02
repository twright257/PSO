__author__ = 'Toasty'
import numpy as np
import random

class ParticleSwarm:

    def __init__(self, dataset, popSize, clusters, tune1, tune2):
        self.dataset = dataset
        self.popSize = popSize
        self.numClusters = clusters
        self.tune1 = tune1
        self.tune2 = tune2
        self.cluster()

    #select population from dataset
    def select_pop(self):
        population = []
        for p in range(self.popSize):
            particle = []
            for c in range(self.numClusters):
                randIndex = random.randint(0, len(self.dataset) - 1)
                particle.append(self.dataset[randIndex])
            population.append(particle)
        return population

    #run PSO
    def cluster(self):
        pop = self.select_pop() #pick population
        pBest = pop #initialize best particles
        pBestFitness = [1000] * len(pop)
        velocity = np.zeros((len(pop), self.numClusters, len(self.dataset[0])))
        gBest = []
        gBestFitness = 1000
        for runs in range(100):
            for index, particle in enumerate(pop):
                clust = []
                for i in range(self.numClusters):
                    c = []
                    clust.append(c)
                for d in self.dataset:  #set particle to cluster with nearest centroid
                    closest = [1000, 0]
                    for i in range(len(particle)):
                        dist = np.linalg.norm(d - particle[i])
                        if dist < closest[0]:
                            closest = [dist, i]
                    clust[closest[1]].append(d)
                fitness = 0
                for i in range(len(clust)): #evaluate fitness of particle
                    for c in clust[i]:
                        fitness += (np.linalg.norm(c - particle[i]) / len(clust[i]))
                fitness = fitness / len(clust)
                if fitness < gBestFitness:  #set global best
                    gBestFitness = fitness
                    gBest = particle
                if fitness < pBestFitness[index]: # set personal best
                    pBestFitness[index] = fitness
                    pBest[index] = particle
                #update velocity and positon
                pBestMinus = self.p_subtract(pBest[index], particle)
                gBestMinus = self.p_subtract(gBest, particle)
                pBestMinus = self.p_mult(pBestMinus, self.tune1)
                gBestMinus = self.p_mult(gBestMinus, self.tune2)
                gBestMinus = np.add(pBestMinus, gBestMinus)
                velocity[index] = np.add(gBestMinus, velocity[index])
                pop[index] = self.p_add(particle, velocity[index])


    #method for adding particles to each other
    def p_add(self, p1, p2):
        result = []
        for i in range(len(p2)):
            answer = np.add(p1[i], p2[i])
            result.append(answer)
        return result

    #mthod for subtracting particles from each other
    def p_subtract(self, p1, p2):
        result = []
        for i in range(len(p2)):
            answer = np.subtract(p1[i], p2[i])
            result.append(answer)
        return result


    #method for particle multiplication
    def p_mult(self, particle, param):
        rVector = np.zeros((len(particle), len(particle[0])))
        for i, v in enumerate(rVector):
            for j in range(len(particle[0])):
                v[j] = random.uniform(0, param)
                v[j] = v[j] * particle[i][j]
        return rVector

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
            variables = np.array(variables)
            dataset.append(variables)

    pso = ParticleSwarm(dataset, 10, 3, 1.0, 1.0)


if __name__ == '__main__':
    main()

