import numpy as np
import nashpy as nash
from typing import Optional, List, Tuple, Any

class Model:

    def __error_check(self,
                      game_matrix: Any,
                      initial_population: Any,
                      sampling_frequency: Any,
                      mutation_matrix: Any               
        ) -> Tuple[bool, Exception]:
        
        #Check input types:
        if not isinstance(game_matrix, np.ndarray):
            return (False, ValueError("`game_matrix` must be of type `numpy.ndarray`."))

        if not isinstance(initial_population, np.ndarray):
            return (False, ValueError("`initial_population` must be of type `numpy.ndarray`."))

        if not isinstance(sampling_frequency, int):
            return (False, ValueError("`sampling_frequency` must be of type `int`."))

        if not isinstance(mutation_matrix, np.ndarray) or mutation_matrix is not None:
            return (False, ValueError("`mutation_matrix` must either be of type `np.ndarray` or `None`."))
        
        #Sampling frequency must be positive:
        if sampling_frequency < 0:
            return (False, ValueError("`sampling_frequency` must either be a positive integer."))

        #Check whether game_matrix dimensions agree:
        game_num_of_rows, game_num_of_cols = game_matrix.shape

        if game_num_of_rows != game_num_of_cols:
            return (False, ValueError("`game_matrix` dimensions do not agree."))

        #Check whether initial_pop dimensions agree with game_matrix dimensions and are proper:

        pop_num_of_rows, pop_num_of_cols = initial_population.shape

        if pop_num_of_rows != 1 or pop_num_of_cols != game_num_of_rows:
            return (False, ValueError("`initial_pop` dimensions are not proper."))

        #If mutation_matrix is not None, check whether mutation_matrix dimensions agree with game_matrix:

        if mutation_matrix is not None:
            mut_num_of_rows, mut_num_of_cols = mutation_matrix.shape

            if mut_num_of_rows != mut_num_of_cols or mut_num_of_rows != game_num_of_rows or mut_num_of_cols != game_num_of_cols:
                return (False, ValueError("`mutation_matrix` dimensions are not proper."))

        return (True, None)

    def __init__(self, game_matrix: np.ndarray,
                 initial_population: np.ndarray,
                 sampling_frequency: int,
                 mutation_matrix: Optional[np.ndarray] = None,
                 run_checks: Optional[bool] = False
    ) -> None:

        if run_checks is True:
            report = self.__error_check(game_matrix,initial_population,sampling_frequency, mutation_matrix)

            if report[0] is False:
                raise report[1]

        #Initialize class properties;
        self._sampling_frequency: int = sampling_frequency
        self._timepoints: np.ndarray = np.linspace(0, 1, sampling_frequency)
        self._dimension: int = game_matrix.shape[0]
        self._initial_population: np.ndarray = initial_population
        self._game_matrix: np.ndarray = game_matrix

        if mutation_matrix is None:
            self._mutation_matrix = np.identity(self._dimension)
        else:
            self._mutation_matrix = mutation_matrix

        self._game: nash.Game = nash.Game(self._game_matrix)
        self._replicator_dynamics: np.ndarray = self._game.replicator_dynamics(y0 = self._initial_population,
                                                    timepoints=self._timepoints,
                                                    mutation_matrix=self._mutation_matrix)
        
        #No need to initialize these more than once;
        freq_range: np.ndarray = np.arange(0, self._sampling_frequency)
        dim_range: np.ndarray = np.arange(0, self._dimension)

        #Needed for Visualization
        self._population_evolution: List = [] 
        self._derivative_evolution: List = []
        self._fitness_evolution:    List = []
        self._avg_fitness_evolution: List = []        

        derivatives_matrix: List = []

        fitness_matrix: List = []

        for i in freq_range:
            
            #Calculate derivatives, fitness and average fitness 
            #Theory for this calculation: https://nashpy.readthedocs.io/en/stable/text-book/replicator-dynamics.html#the-replicator-mutation-dynamics-equation
            pop_vector: np.ndarray = self._replicator_dynamics[i]
            fitness: np.ndarray  = game_matrix @ pop_vector
            average_fitness: float = pop_vector.T @ fitness 

            self._avg_fitness_evolution.append(average_fitness)
            fitness_matrix.append(fitness)
            derivatives_matrix.append((fitness * pop_vector) @ self._mutation_matrix -  average_fitness * pop_vector)


        #Needed for Visualization
        for i in dim_range:

            type_i: List  = []
            derivative_i: List  = []
            fitness_i: List = []

            for j in freq_range:
                type_i.append(self._replicator_dynamics[j][i])
                derivative_i.append(derivatives_matrix[j][i])
                fitness_i.append(fitness_matrix[j][i])

            self._population_evolution.append(np.array(type_i))
            self._derivative_evolution.append(np.array(derivative_i))
            self._fitness_evolution.append(np.array(fitness_i))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def sampling_frequency(self) -> int:
        return self._sampling_frequency

    @property
    def initial_population(self) -> np.ndarray:
        return self._initial_population
    
    @property
    def game_matrix(self) -> np.ndarray:
        return self._game_matrix
    
    @property
    def mutation_matrix(self) -> np.ndarray:
        return self._mutation_matrix
    
    @property
    def game(self) -> nash.Game:
        return self._game
    
    @property
    def replicator_dynamics(self) -> np.ndarray:
        return self._replicator_dynamics

    @property
    def population_evolution(self) -> List:
        return self._population_evolution
    
    @property
    def derivative_evolution(self) -> List:
        return self._derivative_evolution

    @property
    def avg_fitness_evolution(self) -> List:
        return self._avg_fitness_evolution

    @property
    def fitness_evolution(self) -> List:
        return self._fitness_evolution

    @property
    def final_population_percentages(self) -> np.ndarray:
        return self._replicator_dynamics[self._sampling_frequency-1]

    @property
    def final_population_derivatives(self) -> List:
        t = []
        for i in range(self._dimension):
            t.append(self._derivative_evolution[i][self._sampling_frequency-1])
        return t
    
    def __extrema(self, list):

        min = list[0][0]
        max = list[0][0]

        for i in np.arange(0, self._dimension):
            for j in np.arange(0, self._sampling_frequency):
                min = min if min <= list[i][j] else list[i][j]
                max = max if max >= list[i][j] else list[i][j]
        
        return (min, max)

    @property
    def population_percentage_extrema(self) -> Tuple[float, float]:
        return self.__extrema(self._replicator_dynamics)

    @property    
    def population_derivative_extrema(self) -> Tuple[float, float]:
        return self.__extrema(self._derivative_evolution)     
    
    @property    
    def fitness_extrema(self) -> Tuple[float, float]:
        return self.__extrema(self._fitness_evolution)     
    