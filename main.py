from typing import List
from collections import defaultdict
from fastapi import FastAPI
from fastapi import FastAPI, Request
from pydantic import BaseModel
import copy
# from english_words import get_english_words_set


import random

app = FastAPI()

class MoveRequest(BaseModel):
    board: str
    moves: str
class MonsterData(BaseModel):
    monsters: List[int]

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/klotski")
async def klotski(move_requests: List[MoveRequest]):
    res_total = []
    def move(direction, r,c):
        return (r+1,c) if direction == 'S' else (r-1,c) if direction == 'N' else (r,c+1) if direction == 'E' else (r,c-1)

    for board_a, moves_a in move_requests:
        blocks = defaultdict(list)
        block_move = {}
        board,moves = board_a[1], moves_a[1]

        for r in range(5):
            for c in range(4):
                cur = board[(r*4) + c]
                blocks[cur].append((r,c))
                if cur not in block_move:
                    block_move[cur] = (0,0)
        
        l = 0

        while l != len(moves):
            block = moves[l]
            d = moves[l+1]
            cur_r,cur_c = block_move[block]
            block_move[block] = move(d,cur_r,cur_c)
            l+=2
            
        res = [['@' for c in range(4)] for r in range(5)]
        
        for block in blocks:
            if block == "@":
                continue
            for r,c in blocks[block]:
                new_r,new_c = block_move[block]
                res[r+new_r][c+new_c] = block
                
        for r in range(len(res)):
            res[r] = "".join(res[r])
        
        res_total.append("".join(res))

    return res_total    

@app.post("/efficient-hunter-kazuma")
async def efficient_hunter_kazuma(data: List[MonsterData]):
    def calculate_efficiency(monsters):
        n = len(monsters)
        dp = [[0] * (n + 2) for _ in range(2)]
        for i in range(n-1, -1, -1):
            for buy in range(2):
                if buy:
                    dp[buy][i] = max(-monsters[i] + dp[0][i+1], dp[1][i+1])
                else:
                    dp[buy][i] = max(monsters[i] + dp[1][i+2], dp[0][i+1])
        return dp[1][0]

    result = []
    for entry in data:
        monsters = entry.monsters
        efficiency = calculate_efficiency(monsters)
        result.append({'efficiency': efficiency})
    
    return result


class DodgeRequest(BaseModel):
    data: str

@app.post("/dodge")
async def dodge(dodge_request: DodgeRequest):
    # dodge_request = """.dd\nr*.\n...\n"""

    data = dodge_request.data.strip().split('\n')
    grid = [list(line) for line in data]
    
    # Idea is to dodge the bullet
    print(grid)
    
    grid_tracker = copy.deepcopy(grid)
    for i in range(len(grid_tracker)):
        for j in range(len(grid_tracker[i])):
            grid_tracker[i][j] = []
    print(grid)
    m = len(grid)
    n = len(grid[0])
    
    def adjust_grid(i, j, direction):
        
        if direction == "u":
            d = 0
            while i >= 0:
                grid_tracker[i][j] += [(d, direction)]
                i -= 1
                d += 1
        elif direction == "d":
            d = 0
            while i < m:
                grid_tracker[i][j] += [(d, direction)]
                i += 1
                d += 1
        elif direction == "l":
            d = 0
            while j >= 0:
                grid_tracker[i][j] += [(d, direction)]
                j -= 1
                d += 1
        elif direction == "r":
            d = 0
            while j < n:
                grid_tracker[i][j] += [(d, direction)]
                j += 1
                d += 1
            
         
    row, col = 0, 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            
            if grid[i][j] in ['l', 'r', 'u', 'd']:
                adjust_grid(i, j, grid[i][j])
            elif grid[i][j] == '*':
                row, col = i, j
    
    
    def isSafe(i, j, time):
        
        if not grid_tracker[i][j]:
            return True
        
        
        
        if i == 0 and j == 0:
            for t, di in grid_tracker[i][j]:
                if (di == "l" or di == "u") and t <= time:
                    return False
        if i == m and j == 0:
            for t, di in grid_tracker[i][j]:
                if (di == "l" or di == "d") and t <= time:
                    return False
        
        if i == 0 and j == n:
            for t, di in grid_tracker[i][j]:
                if (di == "r" or di == "u") and t <= time:
                    return False
        
        if i == m and j == n:
            for t, di in grid_tracker[i][j]:
                if (di == "r" or di == "d") and t <= time:
                    return False
        
        # check for stuck bullets
        # for t, _ in grid_tracker[i][j]:
        safe = True
        for t, _ in grid_tracker[i][j]:
            safe = safe and (time > t)
        
        return safe          
        
        
    def dfs(i, j, actions, time):
        
        for t, _ in grid_tracker[i][j]:
            if time == t:
                return None
            
        if isSafe(i, j, time):
            return actions
        
        # Go left
        left    = None
        right   = None 
        up      = None
        down    = None
        if j - 1 >= 0 and (time + 1, "r") not in grid_tracker[i][j]:
            temp = actions + ["l"]
            left = dfs(i, j - 1, temp, time + 1)
        
        if j + 1 < n and (time + 1, "l") not in grid_tracker[i][j]:
            temp = actions + ["r"]
            right = dfs(i, j + 1, temp, time + 1)
        
        if i - 1 >= 0 and (time + 1, "d") not in grid_tracker[i][j]:
            temp = actions + ["u"]
            up = dfs(i - 1, j, temp, time + 1)
        if i + 1 < m and (time + 1, "u") not in grid_tracker[i][j]:
            temp = actions + ["d"]
            down = dfs(i + 1, j, temp, time + 1)
        
        if right:
            return right
        
        if left:
            return left 
        
        if up:
            return up 
        
        if down:
            return down
        
        return None
    print(grid_tracker)
    # return grid_tracker
    return dfs(row, col, [], 0)
            
    
# @app.post("/wordle-game")
# async def wordle_game(data):
    
    
#     web2lowerset = get_english_words_set(['web2'], lower=True)
    
#     five_words_set = set()
    
#     for word in web2lowerset:
#         if len(word) == 5:
#             five_words_set.add(word)
            

    
#     data = {
#         "guessHistory": ["slate", "lucky", "maser", "gapes", "wages"], 
#         "evaluationHistory": ["?-X-X", "-?---", "-O?O-", "XO-?O", "OOOOO"]
#     }
    
#     guessHistory = data["guessHistory"]
#     evaluationHistory = data["evaluationHistory"]
    
#     if not guessHistory:
#         first_guess = random.choice(list(five_words_set))
#         return first_guess

#     else:
        
#         guess = ['', '', '', '', '']
#         banned_letters = set()
#         possible_letters = set()
        
#         for i in range(len(evaluationHistory)):
#             prev_guess  = guessHistory[i]
#             prev_eval   = evaluationHistory[i]
            
#             for i in range(len(prev_eval)):
#                 if prev_eval[i] == 'X':
#                     possible_letters.add(prev_guess[i])
#                 elif prev_eval[i] == 'O':
#                     guess[i] = prev_guess[i]
#                     possible_letters.add(prev_guess[i])
#                 elif prev_eval[i] == '-':
#                     banned_letters.add(prev_guess(i))
#                 elif prev_eval[i] == '?':
#                     continue
        
#         possible_guess = set()
#         for word in five_words_set:
            
#             for i in range(len(word)):
#                 if word[i] in banned_letters:
#                     break
            
            
            
    
    
#     # curr_letters = 






