# coding: utf-8
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from numpy import around as ar
import simpy

class Race:
    def __init__(self, env, length,*animals):
        """Constructor

        env: simpy Environment
        length: float, length of the racetrack
        *animals: list of animals in the race
        """
        pass

    def run(env, name, v0, a = 0):
        """Animal simulation for tortoise and hare
        env: simpy simulation environment
        name: name of the animal
        speed: speed of the animal (m/s)
        """
        a = [env.process(animal.run()) for animal in animals]
        while True:
            t = env.now
            v = v0 + a*t
            #FIXME: need to be distance-based
            print('The ' + name, 'traveled', v*t, 'meters')
            yield env.timeout(1)



#When do they catch up to each other?
#Make it so that the simulation stops when racetrack ends

class Animal:
    #TODO: Deep learning for stamina/speed optimization
    #TODO: Progress bar GUI

    #modify these for rolling starts
    d0 = 0
    v0 = 0
    step_size = .001 # 1 is 1 second

    def __init__(self, env, name, top_speed, base_acc,
                 base_stamina, regen_rate, tactic = 'sprint-rest'):
        self.tactic = tactic #sprint-walk or sprint-rest
        #Animal base stats
        self.name = name
        self.top_speed = top_speed
        self.base_acc = base_acc
        self.base_stamina = base_stamina
        self.regen_rate = regen_rate
        #Instantaneous stats
        self.dist = Animal.d0
        self.speed = Animal.v0
        self.acc = self.base_acc
        self.stamina = self.base_stamina
        #for monitoring & plotting
        self.cycle = 0
        self.dhist = []
        self.vhist = []
        self.ahist = []
        self.shist = []
        #SimPy environment & processes
        self.env = env
        self.move_proc = env.process(self.move())

    def move(self):
        """racing cycle"""
        if self.tactic == 'sprint-walk':
            while True:
                self.cycle += 1
                sprint = self.env.process(self.sprint())
                yield sprint
                walk = self.env.process(self.walk())
                yield walk
        elif self.tactic == 'sprint-rest':
            while True:
                self.cycle += 1
                sprint = self.env.process(self.sprint())
                yield sprint
                rest = self.env.process(self.rest())
                yield rest

    def sprint(self):
        """While the animals still has stamina, it continues to accelerate until
        it reaches its top speed. After 70% of the stamina has been depleted
        the animal starts to decelerate.
        """
        print("{} starts running at {}".format(self.name, self.env.now))
        while self.stamina > 0:
            #stamina calculated from current speed
            speed_ratio = self.speed/self.top_speed
            stamina_chg = -(speed_ratio)*10
            self.stamina += stamina_chg*Animal.step_size

            #accleration calculated from stamina
            stamina_ratio = self.stamina/self.base_stamina
            self.acc = self.base_acc if stamina_ratio > .3 else -(1 - stamina_ratio)
            self.speed = self.speed + self.acc*Animal.step_size

            #Limiting values
            self.acc = 0 if self.speed >= self.top_speed else self.acc
            self.speed = np.clip(self.speed, 0, self.top_speed)
            self.stamina = np.clip(self.stamina, 0, self.base_stamina)

            self.dist += self.speed*Animal.step_size
            self.store_data()

            yield self.env.timeout(Animal.step_size)
        return self.dist

    def rest(self, percent_recov = .5):
        """Animal stops moving to recover stamina.
        percent_recov: 0-1, percent of stamina the animal chooses to recover
        """
        self.speed = 0
        self.acc = 0
        print("{} starts resting at {}".format(self.name, self.env.now))
        while self.stamina < (self.base_stamina*percent_recov):
            self.stamina += self.regen_rate*Animal.step_size
            self.store_data()
            yield self.env.timeout(Animal.step_size)

    def walk(self, walk_speed = 1.5, percent_recov = 1):
        """Alternatively, the animal can walk for half the regen rate"""
        self.speed = walk_speed
        self.acc = 0
        print("{} starts walking at {}".format(self.name, self.env.now))
        while self.stamina < (self.base_stamina*percent_recov):
            self.stamina += (self.regen_rate*Animal.step_size)/2
            self.store_data()
            yield self.env.timeout(Animal.step_size)

    def store_data(self):
        """Stores information in lists"""
        self.dhist.append(self.dist)
        self.vhist.append(self.speed)
        self.ahist.append(self.acc)
        self.shist.append(self.stamina)

    def plot(self, unit = 'seconds'):
        """Plots the animals history with its statistics"""
        ptext = "top speed: {} m/s, acceleration: {} m/s^2, stamina: {} units, recovery = {} units/s"
        stats = ptext.format(self.top_speed, self.base_acc, self.base_stamina, self.regen_rate)
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle("{}'s performance statistics".format(self.name))
        fig.text(0.5, 0.93, stats, ha = 'center', va = 'top')

        tx = Animal.make_time_axis(self.dhist, unit)
        axs[0].plot(tx, self.dhist, label = 'distance')
        axs[0].legend(loc = 'lower left')
        axs[1].plot(tx, self.vhist, label = 'speed')
        axs[1].plot(tx, self.ahist, label = 'acceleration')
        axs[1].legend(loc = 'lower left')
        axs[2].plot(tx, self.shist, label = 'stamina')
        axs[2].legend(loc = 'lower left')
        #plt.xlabel(unit)

        rtext = "Results: {} sprint-rest cycles, distance: {} m, average speed: {} m/s"
        results = rtext.format(self.cycle, ar(self.dhist[-1], 2),
                               ar(np.mean(self.vhist), 2))
        fig.text(0.5, 0.02, results, ha = 'center', va = 'bottom')
        plt.show()

    @classmethod
    def make_time_axis(cls, series, unit):
        """Helper function for plotting with time axes"""
        ss = cls.step_size
        if unit == 'seconds':
            tx = np.arange(0, len(series)*ss, ss)
        elif unit == 'minutes':
            tx = np.arange(0, (len(series)*ss)/60, ss/60)
        elif unit == 'hours':
            tx = np.arange(0, (len(series)*ss)/3600, ss/3600)
        return tx


def main():
    #peter_parker = Animal('Peter Parker', 70, 10, 1000, 500)
    env = simpy.Environment()
    cow = Animal(env, 'cow', 11.18, 2, 200, 10, tactic = 'sprint-walk')
    #jaguar = Animal(env, 'jaguar', 33.5, 9, 50, 5)
    env.run(until=200)

    # for i in range(2):
    #     cow.move()
    #jaguar.plot()
    cow.plot()

if __name__ == '__main__':
    main()
