# Introduction

This assignment involves implementing a Game Playing AI agent. The goal is to learn how to make decisions in domains with a large space of possibilities while also taking into account the actions of the other agents in a time bound manner.  

The setting we will consider is a game called Havannah (described below). This is a two player moves in which each player tries to create certain structures on a hexagonal board.  The decision-making task is to determine what is a “good” move providing you higher chances of succeeding. While deciding the next move you need to consider the state (of the board and what actions the other player has taken) while considering a time budget. 

Developing an algorithm for this decision making task requires both thought and iterative implementation. Hence, it is recommended to engage with the assignment early. 

# Problem Statement

## Havannah

Havannah is a board game played on a hexagonal grid of varying sizes. Two players, in alternate turns, place their coloured pieces on the hexagonal cells of the board, aiming to achieve one of the three winning conditions (explained below) with an unbroken chain of their pieces. The game continues until one player successfully meets one of these conditions and wins the game.

A board looks like the following, and in this state, players 1 and 2 have kept their pieces at (2,3) and (3,4) respectively, and player 1 is next to move : 

![image](images/Screenshot 2024-09-06 at 11.22.57 AM.png)

## Winning Criteria

A win is declared when a player is successful in making any of the following three structures described below :

Bridge

![Screenshot 2024-08-31 at 2.29.31 AM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/17c8da84-ab62-453b-a9a6-e2529d82f67c/f1670ac4-b1f1-463b-9794-094c623d515b/fabef0d7-5705-47ab-aed5-c13881ce6cb0.png)

Fork

![Screenshot 2024-08-31 at 2.30.30 AM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/17c8da84-ab62-453b-a9a6-e2529d82f67c/2f05fffb-f8d0-4af1-946a-0c36da90e84f/e05b925c-80f9-459e-80c9-f2355eb76097.png)

Ring

![Screenshot 2024-08-31 at 2.30.06 AM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/17c8da84-ab62-453b-a9a6-e2529d82f67c/d6b0a96e-7e8b-48a4-ac1b-9c87df87c76b/46d5714d-a5e6-45a2-ac4c-d88b7db4be0f.png)

- A **Fork** in Havannah occurs when three (or more) different edges of the board are connected through a path of adjacent, same-coloured cells. A cell is considered part of an edge if it lies on the boundary of the board but is not a corner. Importantly, corner cells are not part of any edge. The figure illustrates the yellow player creating a fork by connecting the edge cells at (0,5), (2,6), and (5,4). In this example, the yellow player successfully connects three edge cells on different edges through a continuous path.
- A **Bridge** is a path (a sequence of adjacent, same-coloured cells) that connects any two corners of the board. Note that there are 6 corners on any given board. The figure above illustrates an example where the red player creates a bridge connecting the corners at coordinates (6,3) and (0,0). It is important to note that (1,0) is not a corner but an edge.
- A **Ring** is defined as a closed, cyclical path of same-coloured cells which must enclose one or more cells inside it. Note that the cell(s) inside the enclosure can be either the player’s or the opponent's or an empty block.

## End of the game

The game ends when one of the players is able to complete any of the three structures or the time budget for one of the players is exhausted.  In the latter case, the agent whose time budget is not exhausted is considered the winner.

A win will receive 1 point, a loss receives 0 points. In the case of a draw, a partial credit ( < 1) is awarded based on the time remaining relative to the opponent agent’s time remaining

Across all our evaluations, this scheme will award higher marks to agents that give winning results faster.

# Starter Code

## Code Package and Environment Setup

- The starter code is available on Moodle. The downloaded zip has the following structure. Note that your implementation will only be modifying the files listed in red below. Do not modify any other file(s).

```yaml
A2
├── game.py             ## Game engine - runs the loop/renders/clock/IO
├── helper.py           ## Functions for traversing board/win condition, utils
├── initial_states      ## In case you need to load custom layouts -
│   ├── size* .txt          ## standard sample board representations   
│   ├── custom.txt          ## (optional) if the need be, create a layout like this and load using --start_file flag
├── players             ## Implementations for agents are housed here
│   ├── **ai.py**               ## Your implementation comes here, the ***only*** code file you will submit
│   ├── ai2.py              ## To play ai vs ai games
│   ├── human.py            ## Input either via terminal or by clicking position on window
│   └── random.py           ## Randomly selects one of the valid moves
└── readme.md           ## Instruction
```

- Please setup a **conda environment** with the allowed dependencies for the assignment with the commands provided below. First time conda users can download and install [*Miniconda*](https://docs.conda.io/en/latest/miniconda.html).

```bash
cd A2 
conda create -n aia2 python=3.10 numpy tk
conda activate aia2
```

## Interaction with  Simulator

The game between two players `<player1>` and `<player2>` can be initiated by the following command.  

```bash
python game.py <player1> <player2> [--flag_field flag_value]*
```

| Field | Description | Possible values | Default | Required argument? |
| --- | --- | --- | --- | --- |
| player1, player2 | Agent that plays as player 1  and player 2 (required) | [”ai”, “ai2”, “random”, “human”] | - | Yes |
| time | Time Budget in seconds,  per player, for the entire game | positive integer | 240 | Yes |
| dim | Size of the board | positive integer  $\in$ [4, 10] | 4 | optional |
| mode | Render the GUI or not. If you are working with server that doesn’t have a display, use “server” mode | [“gui”, “server”] | “gui” | optional |
| start_file | Use custom layout through text file (optional) | filename, and file populated with the custom layout | None | optional |

**Self-play (human vs human):**  You can understand the game rules better by playing a few games in human vs human mode. The inputs can be passed either via terminal (eg : “0 , 5”) or just clicking on your desired position. You can initiate a game between two human agents on board dim 6, and a game time of 10 minutes as follows:

```bash
## human vs human 
python game.py human human --dim 6 --time 600
```

**Playing with a random agent:** If you want to test your AI agent with a random agent, then you can initiate a game as follows

```bash
## AI vs Random
python game.py ai random --dim 5 --time 1000
```

**Comparing two AI agents:** You might also compare two of your implementations in [`ai.py`](http://ai.py) and  `ai2.py` , starting from a custom layout `initial_states/custom_layout.txt` as follows: 

```bash
## AI vs AI
python game.py ai ai2 --start_file custom_layout.txt
```

# Implementation Guidelines

## Board representation

A board of size N is represented as a numpy array of size $(2N-1, 2N-1)$. The entries in the numpy array can be one of the following:

- 0 : represents an unfilled location
- 1 : represents a location filled by player 1
- 2 : represents a location filled by player 2
- 3 : Dummy array entries that are not part of the board. These locations are considered invalid (dashed boundaries below), and is not to be filled

The state of the board is represented using the following coordinate frame : 

![Screenshot%202024-08-26%20at%2012.58.41%E2%80%AFAM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/17c8da84-ab62-453b-a9a6-e2529d82f67c/61dd9214-f902-4c60-a73f-5058476cf8f2/0c232c95-a8d3-48bb-9b47-5f1095f2b4df.png)

The topmost layer of cells (highlighted in red on the left) constitutes Row 0.

For an $N$-sized board: 

- The cells of the first row are $(0, 0), (0, 1) ….. (0, 2N-1)$
- The 6 corners of the board, in clockwise order from the top-left corner, are : $(0, 0),\; (0, N-1), \; (0, 2N-1),\; (N-1, 2N-1), \; (2N-1, N-1), \;(N-1, 0)$
- Rows $N$ and onwards are only partially playable. These row may have dummy array elements that are not part of the board. The invalid locations are stored as 3 in the board, and pieces are not to be kept here.

## Other Utility Functions

A few utility functions have been provided in  `havannah/helper.py` . You may use these functions in your implementation. 

1. `fetch_remaining_time(timer, player_num)`: Returns the remaining time for the player `player_num`
2. `get_valid_actions(board)` : Returns all the valid and unfilled positions in the provided state `board`
3. `get_neighbours(dim, pos)` : Returns all the neighbours around a given position `pos` for a `dim` sized board.
4. `get_all_corners(dim)` : Returns all the corner positions for a `dim` sized board.
5. `get_all_edges(dim)`     : Returns all the edge    positions for a `dim` sized board.
6. `get_vertices_on_edge(edge, dim)`  : Returns the positions on an edge of the board for a particular `edge`, and a size `dim` board
7. `get_vertex_at_corner(corner, dim)`  : Returns the position on of a `corner` of the board , and a size `dim` board.
8. `get_edge(pos, dim)`  : Returns the edge on which a position  `pos` lies for a size `dim` board, returns -1 if not an edge piece.
9. `get_corner(pos, dim)`  : Returns the corner on which a position  `pos` lies for a size `dim` board, returns -1 if not a corner piece.
