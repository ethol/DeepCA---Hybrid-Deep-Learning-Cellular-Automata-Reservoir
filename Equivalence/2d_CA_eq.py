import random
import sys
import os
import helpers.helper as helper
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, VonNeumannNeighborhood, CAWindow, EdgeRule

everyNeighbour = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, ],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, ],
                  [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, ],
                  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, ],
                  [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ],
                  [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, ],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                  [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, ],
                  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, ],
                  [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, ],
                  [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ],
                  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                  [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]]

sizexny = 18
lookuptable = {}
initialization = []


class Regular(CellularAutomaton):
    """ Cellular automaton with the evolution rules of conways game of life """

    def __init__(self):
        super().__init__(dimension=[sizexny, sizexny],
                         neighborhood=VonNeumannNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    @staticmethod
    def init_cell_state(cord):  # pylint: disable=no-self-use
        return [initialization[cord[0]][cord[1]]]

    def evolve_rule(self, last_cell_state, neighbors_last_states):
        # new_cell_state = last_cell_state
        zero = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (0, -1))
        one = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (-1, 0))
        two = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (1, 0))
        three = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (0, 1))

        lookupt = np.arange(5)
        lookupt[2] = last_cell_state[0]
        lookupt[0] = zero[0]
        lookupt[1] = one[0]
        lookupt[3] = two[0]
        lookupt[4] = three[0]

        bin = ''.join(str(lookupt))
        return [self.lookup(bin)]

    @staticmethod
    def lookup(neighbourhood):
        return lookuptable[neighbourhood]


class RegularMirror(CellularAutomaton):
    """ Cellular automaton with the evolution rules of conways game of life """

    def __init__(self):
        x = VonNeumannNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS)
        super().__init__(dimension=[sizexny, sizexny],
                         neighborhood=x)

    @staticmethod
    def init_cell_state(cord):  # pylint: disable=no-self-use
        # backwards counting for mirroring initalization
        return [initialization[sizexny - cord[0] - 1][cord[1]]]

    def evolve_rule(self, last_cell_state, neighbors_last_states):
        zero = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (0, -1))
        one = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (-1, 0))
        two = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (1, 0))
        three = self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (0, 1))

        lookupt = np.arange(5)
        lookupt[2] = last_cell_state[0]
        lookupt[0] = zero[0]
        lookupt[1] = two[0]
        lookupt[3] = one[0]
        lookupt[4] = three[0]

        bin = ''.join(str(lookupt))
        return [self.lookup(bin)]

    @staticmethod
    def lookup(neighbourhood):
        return lookuptable[neighbourhood]


if __name__ == "__main__":
    for count in range(0, 1):
        for i in range(0, 32):
            bin = helper.int_to_binary_string(i, 5)

            lookuptable[''.join(str(bin))] = random.randint(0, 1)

            # initialization = np.random.randint(2, size=(sizexny, sizexny))
            initialization = np.array(everyNeighbour)

        regular = Regular()
        CAWindow(cellular_automaton=regular,
                 window_size=(1000, 830)).run(evolutions_per_second=1, last_evolution_step=1)

        # probebly a better way to do this, but this is just to get the cell state at the end of the development
        state = np.random.randint(1, size=(sizexny, sizexny))
        for coordinate, cell in regular.cells.items():
            state[coordinate[1]][coordinate[0]] = cell.state[0]

        regular_mirror = RegularMirror()
        CAWindow(cellular_automaton=regular_mirror,
                 window_size=(1000, 830)).run(evolutions_per_second=1, last_evolution_step=1)

        # probebly a better way to do this, but this is just to get the cell state at the end
        stateMirror = np.random.randint(1, size=(sizexny, sizexny))
        for coordinate, cell in regular_mirror.cells.items():
            stateMirror[coordinate[1]][coordinate[0]] = cell.state[0]
        print(initialization)
        print(state)
        print(np.fliplr(stateMirror))
        print(count, np.array_equal(state, np.fliplr(stateMirror)))
