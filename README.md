# Meltingpot

A ligthweight genetic algorithm for solving real-encoded optimization problems, with constraints too.

Inspired by the natural selection concept, originally developed by [Charles Darwin](https://en.wikipedia.org/wiki/Natural_selection), a genetic algorithm:

1. Creates a random initial population of possible solutions (individuals).
2. Generates a number of new populations, where each generation is created by performing:
    - selection based on fitness evaluation,
    - crossover between parents,
    - mutation of individuals.

When the generation process stops, best individual is chosen as the solution to given problem.

Some caveats:

- The optimization problem is handled as a *minimization* problem, that is Meltingpot tries to find the minimum of given objective function.
- Unlike many common implementations, here fitness function is equal to the objective function: therefore, best individuals have lowest score and vice versa.

### Usage examples

##### Simple case

```
from meltingpot import GeneticAlgorithm

# define the objective function - mandatory
f = lambda x : (x[0]-1)**2+(x[1]-1)**2

# define number of variables - mandatory too
nvars = 2

# init GA
ga = GeneticAlgorithm(f,nvars)

# run and get solution with score
sol,score = ga.run()
```

Functions can be anonymous (lambda) or normal function objects. They must accept a vector of length *nvars* and return a scalar value.

##### Rosenbrock function
Minimize [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) with a global minimum at *(1,1)*. Population set to 100 individuals, fixed mutation operator.

```
from meltingpot import GeneticAlgorithm, Mutation

# Rosenbrock function
a = 1
b = 100
f = lambda x : (a-x[0])**2 + b*(x[1]-x[0]**2)**2
nvars = 2

# define mutation parameters
mutation = Mutation(shrink=0.5,sigma=1)

# define lower/upper boundaries for point coordinates
LB=[-5,-5]
UB=[10,10]

ga = GeneticAlgorithm(f,nvars,LB=LB,UB=UB,pop_size=100,mutation=mutation)
sol,score = ga.run()
```

Run and display results:

```
>>> sol,score = ga.run()
>>> print(sol)
[1.00057427 1.00115261]
>>> print(score)
3.3118979646719515e-07
```

##### Constrained optimization

Minimize the constrained G06 function, which has a minimum at *(14.095, 0.84296)* and is subject to following inequality constraints:

- *-(x[0]-5)^2 - (x[1]-5)^2 + 100 <= 0*
- *(x[0]-6)^2 + (x[1]-5)^2 - 82.81 <= 0*


```
# objective function - G06
f = lambda x : (x[0]-10)**3 + (x[1]-20)**3
nvars = 2

# inequality constraints
g1 = lambda x: -(x[0]-5)**2 - (x[1]-5)**2 + 100
g2 = lambda x: (x[0]-6)**2 + (x[1]-5)**2 - 82.81
ics = [g1,g2]

# define penalty schema
penalty = Penalty(alpha=5,beta=5,C=1000)

# define mutation operator
mutation = Mutation(shrink=1,sigma=5)

# init and run
ga = GeneticAlgorithm(f,nvars,ics=ics,LB=[13,0],UB=[100,100],pop_size=1000,num_iters=100,mutation=mutation,penalty=penalty)
sol,score = ga.run()
```

### Boundaries

Lower and upper boundaries are defined by `LB` and `UB` parameters respectively. They must be a list with length equal to *nvars*.

Boundaries are hard constraints, i.e. the algorithm admits only individuals inside boundaries.

### Selection

Selection is based on Stochastic Universal Sampling (SUS) technique and scales values according to their rank instead of their objective raw score, in order to mitigate effect of score variance. Given ranking of N individuals (where rank 1 is the most fit and N the least), scaled fitness holds 3 basic properties:

1. Given an individual with rank k, it is proportional to 1/sqrt(k).
2. Sum of scaled fitness is equal to the number of candidates needed for the new generation.
3. Scaled values are inversely proportional to raw score, i.e. best individual has the highest scaled value and vice versa.

### Elitism

Define number of elite members to keep with `elites` parameter:

```
ga = GeneticAlgorithm(f,nvars,elites=2)
```
Default value is `elites=2`.

### Crossover

Crossover rate sets the fraction of population which generated by crossover and can be set with `crossover` parameter:

```
ga = GeneticAlgorithm(f,nvars,crossover=0.75)
```

Default value is `crossover=0.6`.

Each pair of parents generates two different individuals with intermediate point method: given parents *p1* and *p2*, child is *p1* + *rand*\*(*p2*-*p1*), being *rand* uniformly distributed in [0,1].

### Mutation
The total fraction of mutation children is equal to *1-crossover*.

Mutation changes individual vectors by adding a zero-mean gaussian random value to its entries; resulting value is clipped to lower/upper bounds. At each iteration sigma is updated according to a `shrink` value and the previous `sigma` value, so that *sigma_k = sigma_k-1 * (1-shrink * k/num_iters)* where *k* is current iteration. Setting *shrink=0* let `sigma` be constant.

Default values are `shrink=1` and `sigma=1`.

### Constraints

Inequality and equality constraints can be passed as function objects using, respectively, `ics` and `ecs` parameters:

```
# objective
f = lambda x: x[0]*x[1]

# equality constraint: x+y=6
g = lambda x: x[0]+x[1]-6
ecs = [g]

# customize the penalty policy
penalty = Penalty(alpha=3,beta=3,C=100)

# set boundaries
LB = [-10,-10]
UB = [10,10]

# init and run
ga = GeneticAlgorithm(f,nvars,ecs=ecs,LB=LB,UB=UB,penalty=penalty)
sol,score = ga.run()
```

Meltingpot handles constraints (both linear and nonlinear) through dynamic penalty functions. Penalty functions aim to replace the constrained optimization problem with an unconstrained problem, which is formed by adding cost terms - penalty functions - to the objective function. A penalty function is equal to zero if the constraint is satisfied, or a positive number if it is violated. Since individuals with lowest score are most fit, adding a positive term decreases their fitness.

The penalty evaluated at the ii-th iteration for individual *x* respect to constraint *cs* is *P = (ii\*C)^a + v^b*, where *C*, *a* and *b* are constant values. For inequality constraints, value *v* is equal to *f(x)* if *cs* is violated or *0* otherwise. For equality constraints, *v* is equal to *abs(f(x))* is *cs* is violated or *0* otherwise.

Default values for penalty are: `C=1`,`alpha=2`,`beta=2`.

### Cataclysms

To detect a premature convergence of individuals towards local solutions, a stall check is performed at each iteration. If stall conditions are verified, a catastrophic event is triggered. Cataclysms keep alive only a small fraction of most fitting individuals and re-generate the remainder of population.

A stall condition is declared if average improvement in best score values over `stall_generations` is less than or equal to `tol_value`.

Counters for mutation and penalty functions are reset, as a new evolution stage would actually imply.

```
from meltingpot import Stall

# define a customized stall function
stall = Stall(tol_value=0.1,stall_generations=10)

# pass stall function
ga = GeneticAlgorithm(f,nvars,stall=stall)

# set fraction of cataclism survivors
ga.survivors = 0.2

```
Default value are `tol_value=0.01` and `stall_generations=5`.
