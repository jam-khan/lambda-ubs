from typing import List, Tuple
from collections import defaultdict
from fastapi import FastAPI
from fastapi import FastAPI, Request
from pydantic import BaseModel
from functools import cache

# from english_words import get_english_words_set


import random

app = FastAPI()

class MoveRequest(BaseModel):
    board: str
    moves: str
class MonsterData(BaseModel):
    monsters: List[int]
class BugFixerRequest(BaseModel):
    bugseq: List[List[int]]
class BugFixerRequest1(BaseModel):
    time: List[int]
    prerequisites : List[List[int]]


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

@app.post("/bugfixer/p2")
async def bug_fixer_p2(data:List[BugFixerRequest]):
    res = []
    for input in data:
        input = input.bugseq
        @cache
        def dp(i, time):
            if i == len(input):
                return 0
            diff,limit = input[i]

            res = float('-inf')

            if time+diff <= limit:
                res = max(res, 1+dp(i+1, time+diff))
            res = max(res, dp(i+1, time))

            return res
        res.append(dp(0,0))
        
    return res


@app.post("/bugfixer/p1")
async def bug_fixer_p1(data : List[BugFixerRequest1]):
    res = []
    for input in data:
        time, pre = input.time, input.prerequisites
        adj_list = defaultdict(list)
        times = defaultdict(list)
        source = [1]*(len(time)+1)

        for i in range(len(time)):
            times[i+1] = time[i]
        
        for a,b in pre:
            adj_list[a].append(b)
            source[b] = 0
        
        max_cost = {}
        cur_res = [float('-inf')]
        
        def dfs(n, cost):
            if n in max_cost:
                max_cost[n] = max(max_cost[n],cost)
                cur_res[0] = max(cur_res[0], max_cost[n])
                return 

            max_cost[n] = cost

            for neigh in adj_list[n]:
                dfs(neigh, cost+times[neigh])
        

        for i in range(1,len(source)):
            if source[i] == 1:
                dfs(i, times[i])
                cur_res[0] = max(cur_res[0], times[i])
        
        res.append(cur_res[0])
    return res




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






