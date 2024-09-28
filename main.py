from typing import List, Tuple
from collections import defaultdict, deque
from fastapi import FastAPI
from fastapi import FastAPI, Request
from pydantic import BaseModel
import copy
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

class InterpreterData(BaseModel):
    expressions: List[str]

class DigitalColony(BaseModel):
    generations: int
    colony:  str

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
    print("hit")
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

        def bfs(q):
            while q:
                pop,cost = q.popleft()
                max_cost[pop] = max(max_cost[pop], cost) if pop in max_cost else cost
                cur_res[0] = max(cur_res[0], max_cost[pop])

                for neigh in adj_list[pop]:
                    q.append((neigh, cost+times[neigh]))

        q = deque()
        for i in range(1,len(source)):
            if source[i] == 1:
                q.append((i, times[i]))
        
        bfs(q)
        
        res.append(cur_res[0])
    return res

@app.post("/digital-colony")
async def digital_colony(data : List[DigitalColony]):
    res = []
    for input in data:
        g, c = input.generations, input.colony
        for i in range(g):
            pairs = deque([])
            total = 0
            for j in range(len(c)):
                total += int(c[j])
                if j == len(c)-1:
                    continue
                cur,nex = int(c[j]),int(c[j+1])
                if cur == nex:
                    pairs.append('0')
                elif cur == 1:
                    pairs.append(str(10- abs(cur-nex))[-1])
                else:
                    pairs.append(str(abs(cur-nex))[-1])
            new_c = ""

            for i in range(len(c)):
                if i == len(c)-1:
                    continue
                new_c+= c[i]
                pop = pairs.popleft()
                new_c += pop
            c = new_c
        res.append(c)
    return res



@app.post("/lisp-parser")
async def bug_fixer_p2(data:InterpreterData):
    def solve(data):
        output = []  
        codes = data.expressions
        symbols = {}  
        def error(line):
            return {"output" : output + [f"ERROR at line {line + 1}"]}
        def checkType(val, desired):
            val = symbols.get(val, val)
            if desired == int:
                desired = (int, float)
            return isinstance(val, desired)
        
        for i, line in enumerate(codes):
            index = 0
            stack = []  
            print(codes)
            while index < len(line):
                curr = line[index]

                if curr == '(':
                    stack.append('(')
                    index += 1 

                elif curr == ')':
                    temp = []
                    while stack and stack[-1] != '(':
                        temp.append(stack.pop(-1)) 
                    stack.pop(-1)

                    temp = list(reversed(temp)) 
                    op = temp[0]  
                    args = temp[1:]
                    size = len(args)
                    if op == "puts":
                        if size != 1 or not checkType(args[0], str):
                            return error(i)
                        output.append(symbols.get(args[0], args[0]))  # Output the value of the argument (symbol or literal)
                        stack.append(None)
                    elif op == "set":
                        if size != 2 or args[0] in symbols:
                            return error(i)
                        
                        stack.append(None)

                        symbols[args[0]] = symbols.get(args[1], args[1])  # Set a variable in the 'symbols' dictionary
                    elif op == "lowercase":
                        if size != 1 or not checkType(args[0], str):
                            return error(i)
                        stack.append(symbols.get(args[0], args[0]).lower())  # Convert the argument to lowercase and push to stack
                    elif op == "uppercase":
                        if size != 1 or not checkType(args[0], str):
                            return error(i)
                        stack.append(symbols.get(args[0], args[0]).upper())  # Convert the argument to uppercase and push to stack
                    elif op == "concat":         
                        if size != 2 or not checkType(args[0], str) or not checkType(args[1], str):
                            return error(i)
                        stack.append(symbols.get(args[0], args[0]) + symbols.get(args[1], args[1]))
                    elif op == "replace":
                        if size != 3 or not checkType(args[0], str) or not checkType(args[1], str) or not checkType(args[2], str):
                            return error(i)
                        source = symbols.get(args[0], args[0])
                        target = symbols.get(args[1], args[1])
                        replacement = symbols.get(args[2], args[2])
                        result = source.replace(target, replacement)
                        stack.append(result) 
                    elif op == "substring":
                        if size != 3 or not checkType(args[0], str) or not checkType(args[1], int) or not checkType(args[2], int) or args[1] < 0 or args[2] < 0:
                            return error(i) 
                        s = symbols.get(args[0], args[0])
                        l = symbols.get(args[1], args[1])
                        r = symbols.get(args[2], args[2])
                        if l < 0 or r < 0 or l >= len(s) or r > len(s) or l > r:
                            return error(i)
                        stack.append(s[l:r])
                    elif op == "add":
                        if size < 2:
                            return error(i)
                        res = 0
                        for num in args:
                            if not checkType(num, int):
                                return error(i)
                            res += symbols.get(num, num)
                        stack.append(res)
                    elif op == "subtract":
                        if size < 2:
                            return error(i)
                        res = args[0]
                        if not checkType(res, int):
                            return error(i)
                        for num in args[1::]:
                            if not checkType(num, int):
                                return error(i)
                            res -= symbols.get(num, num)
                        stack.append(res)
                    elif op == "multiply":
                        if size < 2:
                            return error(i)
                        
                        res = 1
                        for num in args:
                            if not checkType(num, int):
                                return error(i)
                            res *= symbols.get(num, num)
                        stack.append(res)
                    elif op == "divide":
                        if size != 2 or not checkType(args[0], int) or not checkType(args[1], int) or args[1] == 0 :
                            return error(i)
                        stack.append(symbols.get(args[0], args[0]) / symbols.get(args[1], args[1]))
                    elif op == "abs":
                        if size != 1 or not checkType(args[0], int):
                            return error(i)
                        stack.append(abs(symbols.get(args[0], args[0])))
                    elif op == "max":
                        if size < 1:
                            return error(i)
                        res = args[0]
                        if not checkType(res, int):
                            return error(i)
                        for num in args[1::]:
                            if not checkType(num, int):
                                return error(i)
                            res = max(res, symbols.get(num, num))
                        stack.append(res)
                    elif op == "min":
                        if size < 1:
                            return error(i)
                        res = args[0]
                        if not checkType(res, int):
                                return error(i)
                        for num in args[1::]:
                            if not checkType(num, int):
                                return error(i)
                            res = min(res, symbols.get(num, num))
                        stack.append(res)
                    elif op == "gt":
                        if size != 2 or not checkType(args[0], int) or not checkType(args[1], int):
                            return error(i)
                        stack.append(symbols.get(args[0],args[0]) > symbols.get(args[1], args[1]))
                    elif op == "lt":
                        if size != 2 or not checkType(args[0], int) or not checkType(args[1], int):
                            return error(i)
                        stack.append(symbols.get(args[0],args[0]) < symbols.get(args[1], args[1]))
                    elif op == "equal":
                        if size != 2:
                            return error(i)
                        print(args[0] == args[1])
                        stack.append(symbols.get(args[0],args[0]) == symbols.get(args[1], args[1]))
                    elif op == "not_equal":
                        if size != 2:
                            return error(i)
                        stack.append(symbols.get(args[0],args[0]) != symbols.get(args[1], args[1]))
                    elif op == "str":
                        if size != 1:
                            return error(i)
                        res = symbols.get(args[0],args[0])
                        if res == None:
                            res = "null"
                        elif isinstance(res, bool):
                            if res:
                                res = "true"
                            else:
                                res = "false"
                        else:
                            res = str(res)
                        
                        stack.append(res)
                    index += 1 



                else:
                    if curr != ' ':

                        if curr == '\"':  # Handle strings
                            word = ""
                            index += 1  # Skip the opening quote
                            while index < len(line) and line[index] != '\"':
                                word += line[index]
                                index += 1
                            stack.append(word)
                            index += 1  # Skip the closing quote
                        else:
                            word = ""
                            while index < len(line) and line[index] not in " ()":
                                word += line[index]
                                index += 1
                            try:
                                # Try to convert to a number
                                word = float(word) if '.' in word else int(word)
                            except ValueError:
                                pass
                            if word == "null":
                                word = None
                            if word == "true":
                                word = True
                            if word == "false":
                                word = False
                            stack.append(word)
                            continue 
                    else:
                        index += 1

        return {"output": output}
        
    return solve(data)


class DodgeRequest(BaseModel):
    data: str

@app.post("/dodge")
async def dodge(request: Request):
    # dodge_request = """.dd\nr*.\n...\n"""
    raw_text = await request.body()  # Get the raw body as bytes
    decoded_text = raw_text.decode("utf-8")
    
    # Process the decoded text
    data = decoded_text.strip().split('\n')  # Split the text into lines
    grid = [list(line) for line in data]  # Convert each line into a list of characters
    
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
    return {"instructions": dfs(row, col, [], 0)}
            
    
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






