from typing import List, Tuple
from collections import defaultdict, deque
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

class InterpreterData(BaseModel):
    expressions: List[str]

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


@app.post("/lisp-parser")
async def bug_fixer_p2(data:InterpreterData):
    def solve(data):
        output = []  
        codes = data.expressions

        symbols = {}  
        def error(line):
            return f"ERROR at line {line + 1}"
        for i, line in enumerate(codes):
            index = 0
            stack = []  

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
                        if size != 1:
                            return error(i)
                        output.append(symbols.get(args[0], args[0]))  # Output the value of the argument (symbol or literal)
                    elif op == "set":
                        if size != 2:
                            return error(i)
                        symbols[args[0]] = symbols.get(args[1], args[1])  # Set a variable in the 'symbols' dictionary
                    elif op == "lowercase":
                        if size != 1:
                            return error(i)
                        stack.append(symbols.get(args[0], args[0]).lower())  # Convert the argument to lowercase and push to stack
                    elif op == "uppercase":
                        if size != 1:
                            return error(i)
                        stack.append(symbols.get(args[0], args[0]).upper())  # Convert the argument to uppercase and push to stack
                    elif op == "concat":         
                        if size != 2:
                            return error(i)
                        stack.append(symbols.get(args[0], args[0]) + symbols.get(args[1], args[1]))
                    elif op == "replace":
                        if size != 3:
                            return error(i)
                        target = symbols.get(args[0], args[0])
                        old = symbols.get(args[1], args[1])
                        new = symbols.get(args[2], args[2])
                        result = target.replace(old, new)
                        stack.append(result)  # Replace string and push the result to stack
                    elif op == "substring":
                        if size != 3:
                            return error(i)
                        s = symbols.get(args[0], args[0])
                        l = symbols.get(args[1], args[1])
                        r = symbols.get(args[2], args[2])
                        stack.append(s[l:r])
                    elif op == "add":
                        if size < 2:
                            return error(i)
                        res = 0
                        for num in args:
                            res += symbols.get(num, num)
                        stack.append(res)
                    elif op == "subtract":
                        if size < 2:
                            return error(i)
                        res = 0
                        for num in args:
                            res -= symbols.get(num, num)
                        stack.append(res)
                    elif op == "multiply":
                        if size < 2:
                            return error(i)
                        res = 1
                        for num in args:
                            res *= symbols.get(num, num)
                        stack.append(res)
                    elif op == "divide":
                        if size != 2 or args[1] == 0:
                            return error(i)
                        res = args[0]
                        for num in args[1::]:
                            res /= symbols.get(num, num)
                        stack.append(res)
                    elif op == "abs":
                        if size != 1:
                            return error(i)
                        stack.append(abs(symbols.get(args[0], args[0])))
                    elif op == "max":
                        if size < 1:
                            return error(i)
                        res = args[0]
                        for num in args[1::]:
                            res = max(res, symbols.get(num, num))
                        stack.append(res)
                    elif op == "min":
                        if size < 1:
                            return error(i)
                        res = args[0]
                        for num in args[1::]:
                            res = min(res, symbols.get(num, num))
                        stack.append(res)
                    elif op == "gt":
                        if size != 2:
                            return error(i)
                        stack.append(symbols.get(args[0],args[0]) > symbols.get(args[1], args[1]))
                    elif op == "lt":
                        if size != 2:
                            return error(i)
                        stack.append(symbols.get(args[0],args[0]) < symbols.get(args[1], args[1]))
                    elif op == "equal":
                        if size != 2:
                            return error(i)
                        stack.append(symbols.get(args[0],args[0]) == symbols.get(args[1], args[1]))
                    elif op == "not_equal":
                        if size != 2:
                            return error(i)
                        stack.append(symbols.get(args[0],args[0]) != symbols.get(args[1], args[1]))
                    elif op == "str":
                        if size != 1:
                            return error(i)
                        stack.append(str(symbols.get(args[0],args[0])))
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
                            stack.append(word)
                            continue 
                    else:
                        index += 1

        return {"output" : output}
        
    return solve(data)


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






