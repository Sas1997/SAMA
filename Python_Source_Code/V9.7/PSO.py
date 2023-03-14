from Fitness import fitness
import numpy as np
import time
from InputData import InData


class PsoParticle:
    """
    This class represents one particle of the population. a possible solution to the problem
    """

    def __init__(self, position, velocity):
        self.position = position  # the position of the particle
        self.velocity = velocity  # the velocity of the particle
        self.fitness_value = None  # every possible solution has its fitness value.
        self.p_best_position = None
        self.p_best_fitness = None


class PSO:
    """
        This class consists of the methods needed to run the PSO algorithm
    """

    def __init__(self):
        # parameters of the PSO
        self.InData = InData  # Problem data used for fitness function
        self.number_of_variables = InData.number_of_variables
        self.maximum_iteration = 10
        self.population_size = 20
        self.current_iteration = 0

        self.g_best_position = None  # the particle that represents the best solution of the optimization process
        self.g_best_fitness = None
        self.particles = None  # the population of particles

        self.w_initial = 0.9  # Inertia Weight - initial value
        self.w_final = 0.2  # Inertia Weight - final value
        self.c_pbest_initial = 2  # Personal Learning Coefficient - initial value
        self.c_pbest_final = 0.8  # Personal Learning Coefficient - final value
        self.c_gbest_initial = 0.6  # Global Learning Coefficient - initial value
        self.c_gbest_final = 2  # Global Learning Coefficient - final value

        # Variables limits
        self.minimum_var_range = InData.minimum_var_range
        self.maximum_var_range = InData.maximum_var_range
        # Velocity Limits

        self.maximum_velocity = 0.3 * (InData.maximum_var_range - InData.minimum_var_range)
        self.minimum_velocity = -1 * self.maximum_velocity

    # The __str__() method returns a human-readable, or informal,
    # string representation of an object. This method is called by the built-in print()
    # , str() , and format() functions.
    def __str__(self):
        return f'{self.g_best_position} , {self.g_best_fitness}'

    def initial_population(self):
        """
        Creates initial population.
        """

        init_population = []
        for k in range(self.population_size):
            x = np.random.uniform(self.minimum_var_range, self.maximum_var_range, (1, self.number_of_variables))
            v = np.random.uniform(self.minimum_velocity, self.maximum_velocity, (1, self.number_of_variables))
            init_population.append(PsoParticle(x, v))

        return init_population

    def adaptive_parameters_cal(self):
        """
         this method is calling during iterations to update values of inertia weight (W) and C1  and C2
        """
        w = self.w_initial + (self.w_final - self.w_initial) * self.current_iteration / self.maximum_iteration

        c_pbest = self.c_pbest_initial + (self.c_pbest_final - self.c_pbest_initial) * self.current_iteration / self. \
            maximum_iteration

        c_gbest = self.c_gbest_initial + (self.c_gbest_final - self.c_gbest_initial) * self.current_iteration / self. \
            maximum_iteration

        return w, c_pbest, c_gbest

    def position_limiter(self, particle):
        """
        to modify particles positions to be in the boundaries
        """
        particle.position = np.minimum(np.maximum(particle.position, self.minimum_var_range), self.maximum_var_range)

    def velocity_limiter(self, particle):
        """
        to modify particles velocities to be in the boundaries
        """
        particle.velocity = np.minimum(np.maximum(particle.velocity, self.minimum_velocity), self.maximum_velocity)

    def particles_evaluation(self):
        """
        to calculate fitness values of each particle
        """
        for particle in self.particles:
            particle.fitness_value = fitness(particle.position)

    def p_best_update(self):
        """
               to update personal best experience of every particle
        """
        for particle in self.particles:
            if particle.p_best_fitness:
                if particle.fitness_value < particle.p_best_fitness:
                    particle.p_best_fitness = particle.fitness_value
                    particle.p_best_position = particle.position
            else:
                particle.p_best_fitness = particle.fitness_value
                particle.p_best_position = particle.position

    def g_best_update(self):
        """
               to update global best experience of every particle
        """
        for particle in self.particles:
            if self.g_best_fitness:
                if particle.fitness_value < self.g_best_fitness:
                    self.g_best_fitness = particle.fitness_value
                    self.g_best_position = particle.position
            else:
                self.g_best_fitness = particle.fitness_value
                self.g_best_position = particle.position

    def velocity_position_update(self):
        """
               to update the velocity and position of every particle
        """
        # adaptive parameters calculations during each iteration
        w, c_pbest, c_gbest = self.adaptive_parameters_cal()
        for particle in self.particles:
            randp = np.random.uniform(0, 1, (1, self.number_of_variables))
            randg = np.random.uniform(0, 1, (1, self.number_of_variables))
            particle.velocity = w * particle.velocity + c_pbest * np.multiply(randp,
                                                                              particle.p_best_position - particle.position
                                                                              ) \
                                + c_gbest * np.multiply(randg, self.g_best_position - particle.position)

            self.velocity_limiter(particle)

            particle.position = particle.position + particle.velocity

            self.position_limiter(particle)

    def run(self):

        start_time = time.time()
        self.particles = self.initial_population()
        best_values = []
        while self.current_iteration < self.maximum_iteration:
            print(f"Iteration: {self.current_iteration}")

            # evaluating the particles
            self.particles_evaluation()

            # updating the Pbests
            self.p_best_update()

            # updating the Gbests
            self.g_best_update()

            # updating the velocities and positions
            self.velocity_position_update()

            # incrementing iterations
            self.current_iteration += 1

            best_values.append([self.current_iteration, self.g_best_fitness])
            print(f"Current best solution: {self}")
            print()

        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)  # The divmod() method takes two numbers as arguments and
        # returns their quotient and remainder in a tuple.
        minutes, seconds = divmod(rem, 60)

        iterations, values = zip(*best_values)  # we are extracting iterations and respective
        # values by zip function, And the asterisk before the iterable means that
        # we can give any number of arguments.
        from Plot_Methods import plot_convergence
        plot_convergence(iterations, values)
        print()
        print(f"Final best solution:{self}")
        print('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

        print("------Results------")
        from Results import results
        from Results import results
        from Utility_results import Utility_results
        Psell = results(self.g_best_position)
        Utility_results(Psell)
