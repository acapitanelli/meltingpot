from random import sample as random_sample
from collections import namedtuple
from .penalties import Penalty
import numpy as np

Fitness = namedtuple('Fitness',['vals','cumsum'])

class Stall:
    """ a stall is declared if average change of best scores over last
    stall_generations is less than or equal to tol_value """

    __slots__ = ['tol_value','stall_generations','scores']

    def __init__(self,tol_value=0.001,stall_generations=10):
        # tolerance value
        self.tol_value = tol_value
        # number of generations over which avg change is evaluated
        self.stall_generations = stall_generations
        # last best scores
        self.scores = []

    def check(self):
        """ check for stall conditions """

        if len(self.scores)==self.stall_generations:
            is_stall = -np.mean(np.diff(self.scores))<=self.tol_value
        else:
            is_stall = False
        return is_stall

    def update(self,score):
        """ update list of last best scores """

        self.scores.append(score)
        if len(self.scores)>self.stall_generations:
            self.scores.pop(0)
        return

    def reset(self):
        self.scores = []

class Elites:
    """ elite individuals """

    __slots__ = ['members','score']

    def __init__(self,members,score):
        self.members = members
        self.score = score

class Mutation:
    """ mutation operator """

    __slots__ = ['shrink','sigma']

    def __init__(self,shrink=1,sigma=1):
        self.shrink = shrink
        self.sigma = sigma

class GeneticAlgorithm():
    """ main class, implement genetic algorithm """

    def __init__(self,
        objective,nvars,
        pop_size=500,
        num_iters=250,
        crossover=0.6,
        elites=2,
        ics=None,
        ecs=None,
        mutation=None,
        penalty=None,
        stall=None,
        LB=None,
        UB=None):

        # mandatory args
        self.objective = objective
        self.nvars = nvars

        # optional args
        self.ics = ics or []
        self.ecs = ecs or []
        self.pop_size = pop_size
        self.num_iters = num_iters
        self.elite_count = elites

        # fitness scaled values
        self.scaled_fitness = self._init_scaled_fitness()

        # crossover
        if not (crossover>=0 and crossover<=1):
            raise ValueError('Crossover rate must be in range [0,1]')

        num_crossover = round(pop_size*crossover)
        if (num_crossover%2): num_crossover-=1
        self.num_crossover = num_crossover

        # boundaries
        if LB and len(LB)!=nvars:
            raise ValueError('Lower bounds must have same length than nvars')
        if UB and len(UB)!=nvars:
            raise ValueError('Upper bounds must have same length than nvars')

        self.LB = np.array(LB or [-1 for ii in range(nvars)],dtype='float')
        self.UB = np.array(UB or [1 for ii in range(nvars)],dtype='float')

        # mutation operator
        if mutation:
            self.mutationFcn = mutation
        else:
            initial_sigma = self.UB-self.LB
            self.mutationFcn = Mutation(sigma=initial_sigma)

        # penalty function
        self.penaltyFcn = penalty or Penalty()

        # cataclysm
        self.surviving_rate = 0.1
        self.stallFcn = stall or Stall()

    def run(self):
        """ start optimization process """

        num_iters = self.num_iters

        # generate initial population
        pop = self._init_population()

        # generate elites
        elites = self._init_elites(pop)

        ii = 0
        for _ in range(num_iters):

            # population scores
            sorted_idx,scores = self.eval_objective(pop,ii)

            # update stall function
            self.stallFcn.update(scores[sorted_idx[-1]])

            # checkfor/apply cataclysm
            if self.stallFcn.check():

                # cataclysm!
                pop,sorted_idx,scores = self._trigger_cataclysm(pop,sorted_idx)
                # reset stallFcn
                self.stallFcn.reset()

                # reset counter
                ii = 0

            # feed elites
            elites = self._update_elites(elites,sorted_idx,scores,pop)

            # select candidates for new generation
            selected_pop = self.selection(pop,sorted_idx)

            # crossover
            pop,off_idx = self.apply_crossover(selected_pop)

            # mutation
            pop = self.apply_mutation(pop,off_idx,ii)

            # internal counter
            ii+=1

        # best solution populaiton/elites
        pop_idx,pop_score = self.eval_objective(pop,ii)

        if pop_score[-1]<elites.score[-1]:
            best_sol = pop[pop_idx[-1]]
            best_score = pop_score[-1]
        else:
            best_sol = elites.members[-1]
            best_score = elites.score[-1]

        return best_sol,best_score

    def _init_population(self):
        """ init zero-time population """

        pop_size = self.pop_size
        nvars = self.nvars
        LB = self.LB
        UB = self.UB
        pop = np.empty((pop_size,nvars),dtype=float)
        for ii in range(nvars):
            pop[:,ii] = LB[ii] + (UB[ii]-LB[ii])*np.random.rand(pop_size)

        return pop

    def _init_elites(self,pop):
        """ init elite data """

        pop_size = self.pop_size
        elite_count = self.elite_count

        sorted_idx,scores = self.eval_objective(pop,0)

        best_idx = sorted_idx[pop_size-elite_count:]

        elites = Elites(pop[best_idx],scores[best_idx])

        return elites

    def _update_elites(self,elites,sorted_idx,scores,pop):
        """ update elite members with best individuals """

        pop_size = self.pop_size
        elite_count = self.elite_count

        # best individuals of current generation
        best_idx = sorted_idx[pop_size-elite_count:]
        best_scores = scores[best_idx]
        best_individuals = pop[best_idx]

        # population individuals got best than elites
        if (best_scores[-1]<elites.score[0]) and (best_scores[-1] not in elites.score):

            merged_scores = np.concatenate((best_scores,elites.score))
            merged_individuals = np.concatenate((best_individuals,elites.members))
            idxs = np.argsort(merged_scores)[::-1][-self.elite_count:]

            elites.score = merged_scores[idxs]
            elites.members = merged_individuals[idxs]

        return elites

    def apply_crossover(self, pop):
        """ crossover operation using intermediate point method """

        pop_size = self.pop_size
        nvars = self.nvars
        num_crossover = self.num_crossover
        half_side = int(num_crossover/2)

        # pick unique samples from population
        p_idx = random_sample(range(pop_size),k=num_crossover)

        # two sets of parents
        parents_1 = pop[p_idx[:half_side],:]
        parents_2 = pop[p_idx[half_side:],:]

        # random point
        random_delta = np.random.rand(num_crossover,nvars)

        # generate offspring
        offspring = np.empty((num_crossover,nvars))
        offspring[:half_side,:] = parents_1 + np.multiply(random_delta[:half_side,:],(parents_2-parents_1))
        offspring[half_side:,:] = parents_1 + np.multiply(random_delta[:half_side,:],(parents_2[::-1]-parents_1))

        # update population
        pop[p_idx,:] = offspring

        return pop,p_idx

    def apply_mutation(self,pop,p_idx,ii):
        """ apply adaptive gaussian mutation """

        num_iters = self.num_iters
        pop_size = self.pop_size
        nvars = self.nvars
        UB = self.UB
        LB = self.LB

        # evaluate/update sigma
        mutationFcn = self.mutationFcn
        shrink = mutationFcn.shrink
        prev_sigma = mutationFcn.sigma
        sigma = np.maximum(10**-3,(1-shrink*ii/num_iters)*prev_sigma)
        self.mutationFcn.sigma = sigma

        # select children to mutate -> not crossover-ed
        m_idx = np.setdiff1d(np.arange(pop_size),p_idx,assume_unique=True)

        # mutate
        mutated = pop[m_idx,:] + np.multiply(sigma,np.random.randn(m_idx.size,nvars))

        # clipping to range bounds
        for ii in range(nvars):

            # lower bound
            mutated[:,ii] = np.maximum(LB[ii],mutated[:,ii])
            # upper bound
            mutated[:,ii] = np.minimum(mutated[:,ii],UB[ii])

        pop[m_idx,:] = mutated

        return pop

    def eval_objective(self,pop,ii):
        """ eval fitness value for current population """

        # objective evaluation
        scores = np.apply_along_axis(self.objective,1,pop)

        # evaluate penalty for inequality constraints
        for g in self.ics:
            values = np.apply_along_axis(g,1,pop)
            scores += self.penaltyFcn.eval_ics(values,ii)

        # evaluate penalty for equality constraints
        for g in self.ecs:
            values = np.apply_along_axis(g,1,pop)
            scores += self.penaltyFcn.eval_ecs(values,ii)

        # sort in descending order
        sorted_idx = np.argsort(scores,axis=0)[::-1]

        return sorted_idx,scores

    def _trigger_cataclysm(self,pop,sorted_idx):
        """ run catastrophic event """

        pop_size = self.pop_size
        nvars = self.nvars
        surviving_rate = self.surviving_rate

        num_survivors = round(surviving_rate*pop_size)
        survivors_idx = sorted_idx[pop_size-num_survivors:]

        # init population
        survivors = np.empty((pop_size,nvars),dtype=float)
        for ii in range(nvars):
            survivors[:num_survivors,ii] = pop[survivors_idx,ii]
            survivors[num_survivors:,ii] = self.LB[ii] + self.UB[ii]*np.random.rand(pop_size-num_survivors)

        sorted_idx,scores = self.eval_objective(survivors,0)

        return survivors,sorted_idx,scores

    def selection(self,pop,sorted_idx):
        """ selection of new generation candidates
        selection is based on SUS """

        pop_size = self.pop_size

        # scaled fitness
        fitness = self.scaled_fitness

        # random in [0,1)
        pos = np.random.rand()
        pointers = [pos+ii for ii in range(pop_size)]

        candidate_idxs = []
        for pos in pointers:
            vec = np.nonzero(fitness.cumsum>pos)[0]
            candidate_idxs.append(vec[0])

        # candidates to new generation
        candidates = pop[sorted_idx[candidate_idxs]]

        return candidates

    def _init_scaled_fitness(self):
        """ Init fitness values and fitness cumulative sum. Values are scaled such that
        their sum is equal to the number of individuals needed for next generation. """

        # number of parents = pop_size
        N = self.pop_size

        ranking = np.arange(N)[::-1]+1

        raw = 1/np.sqrt(ranking)
        scaled = N/np.sum(raw)*raw
        cumsum = np.cumsum(scaled)

        return Fitness(scaled,cumsum)
