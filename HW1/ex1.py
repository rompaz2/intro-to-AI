import search
import random
import math
from itertools import product
from scipy.spatial.distance import cityblock

ids = ["315518480", "313306854"]


def floydWarshall(graph, V):
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j]
                                 )
    return dist


class DroneProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """each state will be in the following structure:
        (turn,
        drones - (drone1 - (drone1 name, drone1 loc - (x, y), drone1 carries - (pack1 id, pack2 id)), ...),
        packages - (pack1 - (pack1 name, addressed_to client_name, currently_carried bool, pack1 loc- (x, y)), ...))
        """
        self.cols = len(initial['map'][0])
        self.rows = len(initial['map'])

        self.interruptions = set()
        for i in range(self.rows):
            for j in range(self.cols):
                if initial['map'][i][j] == 'I':
                    self.interruptions.add((i, j))

        graph = self.create_graph(initial['map'])
        dist = floydWarshall(graph, self.rows * self.cols)
        self.distance = self.arrange_dist(dist)

        drones = tuple((d_name, d_loc, ("empty", "empty")) for d_name, d_loc in initial['drones'].items())

        packages_list = []

        self.clients_paths = {}
        self.clients_dropout_locs = {}
        for c_name, c_dict in initial['clients'].items():
            self.clients_paths[c_name] = c_dict['path']
            self.clients_dropout_locs[c_name] = [loc for loc in c_dict['path'] if loc not in self.interruptions]
            for pack in c_dict['packages']:
                packages_list.append((pack, c_name, False, initial['packages'][pack]))

        self.total_packages_to_deliver = len(packages_list)

        packages = tuple(packages_list)

        new_initial = (0, drones, packages)

        search.Problem.__init__(self, new_initial)

    def arrange_dist(self, dist):
        V = self.rows * self.cols
        distances = {}
        for i in range(self.rows):
            for j in range(self.cols):
                distances[(i, j)] = {}
                m = self.flatten_location((i, j))
                for k in range(self.rows):
                    for l in range(self.cols):
                        n = self.flatten_location((k, l))
                        distances[(i, j)][(k, l)] = dist[m][n]
        return distances

    def create_graph(self, map):
        graph = []
        for i in range(self.rows * self.cols):
            graph.append([])
            for j in range(self.rows * self.cols):
                graph[i].append(float('INF'))
        for i in range(self.rows):
            for j in range(self.cols):
                k = self.flatten_location((i, j))
                if i == j:
                    graph[k][k] = 0
                else:
                    if self.check_move((i - 1, j)):
                        l = self.flatten_location((i - 1, j))
                        graph[k][l] = 1
                        if map[i][j] == 'P':
                            graph[l][k] = 1
                    if self.check_move((i + 1, j)):
                        l = self.flatten_location((i + 1, j))
                        graph[k][l] = 1
                        if map[i][j] == 'P':
                            graph[l][k] = 1
                    if self.check_move((i, j - 1)):
                        l = self.flatten_location((i, j - 1))
                        graph[k][l] = 1
                        if map[i][j] == 'P':
                            graph[l][k] = 1
                    if self.check_move((i, j + 1)):
                        l = self.flatten_location((i, j + 1))
                        graph[k][l] = 1
                        if map[i][j] == 'P':
                            graph[l][k] = 1
        return graph

    def check_move(self, target):
        if 0 <= target[0] < self.rows and 0 <= target[1] < self.cols:
            if target not in self.interruptions:
                return True
            else:
                return False
        else:
            return False

    def flatten_location(self, location):
        return self.cols * location[0] + location[1]

    def move(self, turn, drone, packages):
        d_name, d_loc, _ = drone
        xloc, yloc = d_loc
        next_locations = set()
        if yloc >= 1 and (xloc, yloc - 1) not in self.interruptions:
            next_locations.add(("move", d_name, (xloc, yloc - 1)))
        if xloc < self.rows - 1 and (xloc + 1, yloc) not in self.interruptions:
            next_locations.add(("move", d_name, (xloc + 1, yloc)))
        if xloc >= 1 and (xloc - 1, yloc) not in self.interruptions:
            next_locations.add(("move", d_name, (xloc - 1, yloc)))
        if yloc < self.cols - 1 and (xloc, yloc + 1) not in self.interruptions:
            next_locations.add(("move", d_name, (xloc, yloc + 1)))
        return list(next_locations)

    def pickup(self, turn, drone, packages):
        d_name, d_loc, d_carries = drone
        if d_carries[0] != "empty" and d_carries[1] != "empty":
            return []
        can_pickup = set()
        for pack, _, is_carried, p_loc in packages:
            if p_loc == d_loc and not is_carried:
                action = ("pick up", d_name, pack)
                can_pickup.add(action)
        return list(can_pickup)

    def deliver(self, turn, drone, packages):
        d_name, d_loc, d_carries = drone
        pack1, pack2 = d_carries
        if pack1 == "empty" and pack2 == "empty":
            return []
        can_deliver = set()
        for pack, client, _, _ in packages:
            if pack in d_carries and d_loc == self.clients_paths[client][turn % len(self.clients_paths[client])]:
                action = ("deliver", d_name, client, pack)
                can_deliver.add(action)
        return list(can_deliver)

    def wait(self, turn, drone, packages):
        d_name, _, _ = drone
        l = [("wait", d_name)]
        return l

    def remove_bad_permutations(self, permutations):
        """Returns permutations after removing those with conflicting actions.
        Conflict occurs when two drones try to pick up the same package"""
        bad_perm = []
        for perm in permutations:
            perm_check = []
            for drone_action in perm:
                if drone_action[0] == 'pick up':
                    perm_check.append(('pick up', drone_action[2]))
                else:
                    perm_check.append(drone_action)
            perm_check = list(dict.fromkeys(perm_check))
            if len(perm_check) != len(perm):
                bad_perm.append(perm)
        for perm in bad_perm:
            permutations.remove(perm)
        return permutations

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        turn, drones, packages = state
        actions_names = [self.deliver, self.pickup, self.move, self.wait]
        all_drones_actions_dict = {}
        for drone in drones:
            d_name = drone[0]
            all_drones_actions_dict[d_name] = []
            for action in actions_names:
                outcome = action(turn, drone, packages)
                if len(outcome) != 0:
                    all_drones_actions_dict[d_name].extend(outcome)
        possible_actions = list(product(*list(all_drones_actions_dict.values())))
        return tuple(self.remove_bad_permutations(possible_actions))

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        turn, drones, packages = state
        drones_dict = {d_name: {"loc": d_loc, "carries": d_carries} for d_name, d_loc, d_carries in drones}
        packages_dict = {p_name: {"to": p_address, "carried": is_carried, "loc": p_loc}
                         for p_name, p_address, is_carried, p_loc in packages}
        for act in action:
            pack1, pack2 = drones_dict[act[1]]["carries"]
            if act[0] == "move":
                drones_dict[act[1]]["loc"] = act[2]
                if pack1 != "empty":
                    packages_dict[pack1]["loc"] = act[2]
                if pack2 != "empty":
                    packages_dict[pack2]["loc"] = act[2]
            elif act[0] == "pick up":
                inpack = act[2]
                if pack1 == "empty":
                    drones_dict[act[1]]["carries"] = (inpack, pack2)
                else:
                    drones_dict[act[1]]["carries"] = (pack1, inpack)
                packages_dict[inpack]["carried"] = True
            elif act[0] == "deliver":
                unpack = act[3]
                if pack1 == unpack:
                    drones_dict[act[1]]["carries"] = ("empty", pack2)
                else:
                    drones_dict[act[1]]["carries"] = (pack1, "empty")
                del packages_dict[unpack]

        new_drones = \
            tuple((d_name, drones_dict[d_name]["loc"], drones_dict[d_name]["carries"]) for d_name in drones_dict.keys())
        new_packages = \
            tuple((p_name, packages_dict[p_name]["to"], packages_dict[p_name]["carried"], packages_dict[p_name]["loc"])
                  for p_name in packages_dict.keys())
        new_state = (turn + 1, new_drones, new_packages)
        return new_state

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        turn, drones, packages = state
        if len(packages) == 0:
            return True
        return False

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        turn, drones, packages = node.state
        pickups = 0
        n = node
        while n:
            if n.action is not None:
                pickups += sum([1 for act in n.action if act[0] == "pick up"])
            n = n.parent
            if n is not None and n.state[1] == drones and n.state[2] == packages:
                return float('inf')
        h = 0
        p_num = len(packages)
        waiting_drones = set()
        if node.action is not None:
            for act in node.action:
                if act[0] == "wait":
                    waiting_drones.add(act[1])
        drones_locs = set()
        carrying_drones = set()
        for d_name, d_loc, d_carries in drones:
            if d_carries[0] == "empty":
                drones_locs.add(d_loc)
            else:
                carrying_drones.add(d_name)
            if d_carries[1] == "empty":
                drones_locs.add(d_loc)
            else:
                carrying_drones.add(d_name)
        if len(waiting_drones.difference(carrying_drones)) != 0:
            return float("inf")
        for pack in packages:
            p_name, p_address, is_carried, p_loc = pack
            if p_loc in self.interruptions:
                return float('inf')
            if not is_carried and len(drones_locs) != 0:
                h += min({self.distance[p_loc][d_loc] for d_loc in drones_locs})
            h += min({self.distance[p_loc][c_loc] for c_loc in self.clients_dropout_locs[p_address]})
        return (h + (turn + 1) * (pickups + 1) / (self.total_packages_to_deliver + 1)) * p_num

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_drone_problem(game):
    return DroneProblem(game)

