import time
import math
import random
import numpy as np
from pprint import pprint
import sys
from helper import *

corners = set()
# edges = set()

class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.previous_state = None
        self.opponent = 3 - player_number
        self.initialized_globals = False


    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move

        # Parameters
        `state: Tuple[np.array]`
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """
        # Check_immidiate_termination
        win_action = self.can_win(state)
        if win_action:
            return win_action
        # Check if opponent can win
        opp_win_action = self.will_opp_win(state)
        if opp_win_action:
            return opp_win_action
        
        dim = state.shape[0]
        if not self.initialized_globals:
            cor = get_all_corners(dim)
            for c in cor:
                corners.add(c)
            
        # print("State we got")
        # pprint(state.copy())
        opponent_move = None
        if self.previous_state is not None:
            opponent_move = self.identify_opponent_move(self.previous_state, state)
        root = MCTSNode(state, self.player_number, action=opponent_move)
        mcts = MCTS(root, self.player_number)
        best_action = mcts.search()
        self.previous_state = best_action.state
        best_action_to_int = (int(best_action.action[0]), int(best_action.action[1]))
        # print("State Returned")
        # pprint(best_action.state)
        return best_action_to_int
    
    def identify_opponent_move(self, previous_state, current_state):
        for i in range(previous_state.shape[0]):
            for j in range(previous_state.shape[1]):
                if previous_state[i, j] != current_state[i, j] and current_state[i, j] == self.opponent:
                    return (i, j)
        return None
    def can_win(self,state):
        valid_actions = get_valid_actions(state)
        for action in valid_actions:
            new_state = self.make_move(state, action, self.player_number)
            if check_win(new_state, action, self.player_number)[0]:
                return action
        return None
    def will_opp_win(self,state):
        valid_actions = get_valid_actions(state)
        for action in valid_actions:
            new_state = self.make_move(state, action, self.opponent)
            if check_win(new_state, action, self.opponent)[0]:
                return action
        return None
    def make_move(self,state,move,player):
        new_state = state.copy()
        new_state[move] = player
        return new_state
    
    
class MCTSNode:
    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action # action that led to this state by opponent
        self.player = player # player to take action on this state
        self.opponent = 3 - player
        self.children = []
        self.visits = 0
        self.wins = 0

        self.is_terminal = self.check_terminal() #bool : player in parent node has Already Won the game!! 
        self.will_opp_win = self.check_opp_win() #bool : parent node player's opponent will win if we take this action

        self.valid_actions = get_valid_actions(self.state)
        self.heuristic_scores = self.get_heuristic_scores()
        self.unexplored_actions = self.get_unexplored_actions()
        # self.neighbouring_nodes = self.get_neighbouring_nodes()

    def add_child(self, action):
        child_state = self.make_move(self.state, action, self.player)
        child = MCTSNode(child_state, self.opponent, parent=self, action=action)
        self.children.append(child)
        return child

    def make_move(self,state,move,player): # returns new state after making move
        new_state = state.copy()
        new_state[move] = player
        return new_state
    def check_terminal(self):
        if self.parent:
            state = self.make_move(self.parent.state, self.action, self.opponent)
            return check_win(state, self.action, self.opponent)[0]
        return False
    def check_opp_win(self):
        if self.parent:
            state = self.make_move(self.parent.state, self.action, self.player)
            return check_win(state, self.action, self.player)[0]
        return False
    def get_unexplored_actions(self):
        actions = self.valid_actions.copy()
        #sort actions based on heuristic scores
        actions.sort(key=lambda x: self.heuristic_scores[x], reverse=True)
        return actions

    def get_neighbouring_nodes(self):
        if self.parent is None:
            return set()
        neighbouring_nodes = self.parent.neighbouring_nodes.copy()
        neighbours = get_neighbours(self.state.shape[0], self.action)
        if self.action in neighbouring_nodes:
            neighbouring_nodes.remove(self.action)
        neighbouring_nodes.update(neighbours)
        return neighbouring_nodes

    
    def get_heuristic_scores(self):
        heuristic_scores = {}
        state = self.state.copy()
        valid_actions = self.valid_actions
        player = self.player
        for move in valid_actions:
            score = self.combined_heuristic(state, move, player)
            heuristic_scores[move] = score
        return heuristic_scores
    


    def combined_heuristic(self, state,move, player):
        dim  = state.shape[0]
        dimension = (dim+1)/2
        #initialize all to zero
        group_score = locality_score = conn_score = local_reply_score = three_move_score = 0
        locality_score , local_reply_score = \
                self.heuristic_locality(state, move, dim,self.action, player) #self.action is last move
        group_score , conn_score = self.get_group_size(state,move,dim,player) # -1
        # opp_group_score, opp_conn_score = self.get_group_size(state,move,dim,3-player)
        vc_score= self.heuristic_maintain_vc(state,self.action,player,move)
        # group_score -= 1 #To account for the move itself
        group_bonus = 2
        conn_bonus = 20
        three_connector_bonus = 11
        # max_group_score = (3*dimension*dimension-3*dimension+1)//dimension
        # group_score = max(group_score,max_group_score)
        locality_bonus = 2
        local_reply_bonus = 3
        maintain_vc_bonus = 100
        if group_score == 0:
            maintain_vc_bonus = 0

        if self.three_connector_move_heuristic(state, move, dim, player):
            three_move_score = three_connector_bonus

        score= group_score * group_bonus \
            + locality_score * locality_bonus \
            + conn_score * conn_bonus \
            + local_reply_score * local_reply_bonus \
            + vc_score * maintain_vc_bonus +three_move_score #+ opp_group_score * group_bonus + opp_conn_score * conn_bonus
        return score
    #Functions to Caluculate locality Heuristic
    def heuristic_locality(self, state, move, dim, last_move, player):
        neighbours, virtual_conn , not_virtual_conn = self.get_second_layer_connections(state, move, dim)

        locality_score = 0
        local_reply_score = 0

        neighbour_bonus = 3
        virtual_conn_bonus = 2
        virtual_conn_pro_bonus= 5
        not_virtual_conn_bonus = 1
        panic_threat_bonus = 300
        for pos in neighbours:
            if state[pos] == player:
                locality_score += neighbour_bonus
            if pos == last_move:
                local_reply_score += neighbour_bonus
        for pos in virtual_conn:
            if state[pos] == player:
                locality_score += virtual_conn_bonus
                vc_neighbours = get_neighbours(dim, pos)
                i = 0
                for vc_neighbour in vc_neighbours:
                    if vc_neighbour in neighbours:
                        if state[vc_neighbour] == 0:
                            i+=1
                if i == 2:
                    locality_score += virtual_conn_pro_bonus
            if pos == last_move:
                local_reply_score += virtual_conn_bonus
                vc_neighbours = get_neighbours(dim, pos)
                i = 0
                neighbors_under_vc = []
                for vc_neighbour in vc_neighbours:
                    if vc_neighbour in neighbours:
                        if state[vc_neighbour] == 0:
                            neighbors_under_vc.append(vc_neighbour)
                            i+=1
                if i == 2:
                    local_reply_score += virtual_conn_pro_bonus
                    new_state=state.copy()
                    new_state[pos] = 3-player
                    for vc_neighbour in neighbors_under_vc:
                        new_state[vc_neighbour] = 3-player
                        if (check_win(new_state, vc_neighbour, 3-player)[0]):
                            locality_score += panic_threat_bonus
                    
        for pos in not_virtual_conn:
            if state[pos] == player:
                locality_score += not_virtual_conn_bonus
            if pos == last_move:
                local_reply_score += not_virtual_conn_bonus
        return locality_score, local_reply_score


    def get_second_layer_connections(self, board, move, dim):
        umap = {}
        neighbours = get_neighbours(dim, move)
        virtual_connections = []
        neighbours_set = set(neighbours)
        non_virtual_connections = []
        for neighbour in neighbours:
            umap[neighbour] = 1
        for neighbour in neighbours:
            umap[neighbour] = 1
        for neighbour in neighbours:
            #get neighbours of neighbour
            neighbour_neighbours = get_neighbours(dim, neighbour)
            for n in neighbour_neighbours:  
                if n in umap:
                    umap[n]+=1
                else:
                    umap[n] = 1
        for key in umap:
            if umap[key] == 2:
                if key not in neighbours_set:
                    virtual_connections.append(key)
            elif umap[key] == 1:
                non_virtual_connections.append(key)

        return neighbours, virtual_connections, non_virtual_connections
    
    def get_group_size(self, state, move, dim, player):
        # new_board = self.make_move(state, move, player)
        grp_size, connectivity = self.dfs_pro(state, move, dim, player)
        return grp_size, connectivity
    
    def dfs(self, state, move, dim, player):
        visited = set()
        visited_edges = set()
        connectivity = 0
        group=set()
        connectors=set()
        stack = [move]                          
        grp_size = 0
        while stack:
            move = stack.pop()
            if state[move] == player and move not in visited:   
                visited.add(move)
                grp_size += 1
                group.add(move)
                if move in corners: # global variable corners
                    connectivity += 1
                    connectors.add(move)
                else:
                    # if move in edges: # global variable edges
                        edge = get_edge(move,dim)
                        if edge != -1 and edge not in visited_edges:
                            visited_edges.add(edge)
                            connectivity += 1
                            connectors.add(edge)
                neighbours = get_neighbours(dim, move)
                for neighbour in neighbours:
                    if state[neighbour] == player:
                        stack.append(neighbour)
        return group, connectors
    
    def dfs_pro(self, state, move, dim, player):
        neighbours= get_neighbours(dim,move)
        neighbours_under_player = []
        total_group = set()
        total_connectors = set()
        grp_score_needed = False
        conn_score_needed = False
        conn_score = 0
        for neighbour in neighbours:
            if state[neighbour] == player:
                neighbours_under_player.append(neighbour)
        
        for i in range(len(neighbours_under_player)):
            if neighbour not in total_group:
                group, connectors = self.dfs(state, neighbour, dim, player)
                if (i==0):
                    total_group.update(group)
                    total_connectors.update(connectors)
                else:
                    size_total_group = len(total_group)
                    total_group.update(group)
                    new_size_total_group = len(total_group)
                    if new_size_total_group > size_total_group and i!=0:
                        grp_score_needed = True
                    size_total_connectors = len(total_connectors)
                    total_connectors.update(connectors)
                    new_size_total_connectors = len(total_connectors)
                    if new_size_total_connectors > size_total_connectors:
                        conn_score_needed = True
        if (move in corners):
            if len(neighbours_under_player)==0:
                    conn_score_needed = False
                    conn_score = 0.4
            else:
                conn_score_needed = True
                total_connectors.add(move)
        edge = get_edge(move,dim)
        if (edge != -1):
            if edge not in total_connectors:
                if len(neighbours_under_player)==0:
                    conn_score_needed = False
                    conn_score = 0.4
                else:
                    conn_score_needed = True
                    total_connectors.add(edge)

        if grp_score_needed:
            grp_score = len(total_group)
        else:
            grp_score = 0
        if conn_score_needed:
            conn_score = len(total_connectors)
        
        return grp_score, conn_score  

    
    def heuristic_maintain_vc(self,state,last_move,player,move):
        if not last_move:
            return 0
        score=0
        sides =["up","top-right","bottom-right","down","bottom-left","top-left","up","top-right"]
        dim = state.shape[0]
        dimension=(dim+1)/2
        half=1
        if last_move[1]<dimension-1:
            half=-1
        elif last_move[1]==dimension-1:
            half=0
        
        for i in range(6):
            adder1=move_coordinates(sides[i],half)
            adder2=move_coordinates(sides[i+2],half)
            mid_cell_adder = move_coordinates(sides[i+1],half)
            coord1 = (last_move[0]+adder1[0],last_move[1]+adder1[1])
            coord2 = (last_move[0]+adder2[0],last_move[1]+adder2[1])
            mid_cell = (last_move[0]+mid_cell_adder[0],last_move[1]+mid_cell_adder[1])
            if mid_cell==move:
                if is_valid(coord1[0],coord1[1],dim) and is_valid(coord2[0],coord2[1],dim):
                    if state[coord1]==player and state[coord2]==player:
                        score+=1

        return score

    def three_connector_move_heuristic(self, state, move, dim, player):
        neighbours= get_neighbours(dim,move)
        i=0
        edges=[]
        for neighbour in neighbours:
            if state[neighbour]==0:
                if neighbour in corners:
                    i+=1
                edge = get_edge(neighbour,dim)
                if (edge != -1) and (edge not in edges):
                    edges.append(edge)
                    i+=1
        if i==3:
            return True

class MCTS:
    def __init__(self,root,player):
        self.root = root
        self.player = player
        self.opponent = 3 - player
        self.total_simulations = 0
        self.C = 1.41
        self.simulation_limit = 10000
        self.time_limit = 10
    
    def search(self):
        start_time = time.time()
        print(self.player)
        print("Rolling out")
        while time.time() - start_time < self.time_limit:
            # Clear the last print statement
            print(f'\r{self.total_simulations}', end='', flush=True)
            self.total_simulations += 1
            if(self.total_simulations > self.simulation_limit):
                break

            reward = 0
            node, is_terminal = self.select(self.root) #while selecting in each level we expand the node
            #if terminating condition is reached in root node we simply return them
            if node.is_terminal or node.will_opp_win:
                if node.parent == self.root:
                    return node
            if not is_terminal:
                reward = self.simulate(node.state, node.player)
            else:
                reward = 3-node.player
            self.backpropagate(node, reward)

        print()
        return self.get_best_action(self.cmp_visits)
            
    def select(self, node):
        # If the node has no children, expand it
        if not node.children:
            self.expand(node)
            # If the node is terminal, return it immediately
            if node.is_terminal:
                return node, True
            if node.visits == 0:
                return node, False

        best_node = node
        # Traverse the tree until you find a node that either has no children or is terminal
        while best_node.children :
            if best_node.is_terminal:
                return best_node, True
            self.expand(best_node)
            best_node = self.get_best_child(best_node)
        return best_node , False
    
    def expand(self,node):
        if node.unexplored_actions:
            action = node.unexplored_actions.pop()
            child = node.add_child(action)
            return child
        return None
    
    def get_best_child(self,node):
        #check for terminal move 
        best_child = None
        best_score = -1
        for child in node.children:
            if child.is_terminal:
                return child
            if child.will_opp_win:
                return child
            if child.visits == 0:
                return child
            score = self.ucb1(child)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def ucb1(self, node):
        visits = node.visits
        if visits == 0:
            visits =  0.25
        heuristic_score = 0
        if node.parent:
            heuristic_score = node.parent.heuristic_scores[node.action]
        heuristic_bias = heuristic_score/visits
        q_value = node.wins/visits
        exploration_bias = self.C * math.sqrt(math.log(node.parent.visits)/visits)
        return q_value + exploration_bias + heuristic_bias
    
    def simulate(self,state,player):
        current_player = player
        is_win = False
        is_terminal = False
        reward = 0
        while not is_win and not is_terminal:
            valid_actions = get_valid_actions(state)
            if not valid_actions:
                is_terminal = True
                break
            action = random.choice(valid_actions)
            new_state = self.make_move(state, action, current_player)
            move_details = check_win(new_state, action, current_player)
            is_win = move_details[0]
            if is_win:
                reward = current_player
            state = new_state
            current_player = 3 - current_player
        return reward

    def make_move(self,state,move,player):
        new_state = state.copy()
        new_state[move] = player
        return new_state
    
    def backpropagate(self,node,reward):
        while node:
            node.visits += 1
            if reward == 0:
                node.wins += 0.25

            #as at each node we have the player who is going to take action
            if reward != node.player:
                node.wins += 1

            node = node.parent
        
    def get_best_action(self, cmp):
        best_node = None
        best_score = -1
        for child in self.root.children:
            score = cmp(child)
            if score > best_score:
                best_score = score
                best_node = child
        return best_node
    
    def cmp_visits(self, node):
        return node.visits
