from datetime import datetime, timedelta
from typing import List, Tuple
from collections import defaultdict, deque
from fastapi import FastAPI
from fastapi import FastAPI, Request
from pydantic import BaseModel
import copy
from fastapi.responses import JSONResponse
from functools import cache

import random

import pytz
from test_wordle import words
from greens import greens
import copy
from concurrent.futures import ThreadPoolExecutor

reference_words = ["cigar","rebut","sissy","humph","awake","blush","focal","evade","naval","serve","heath","dwarf","model","karma","stink","grade","quiet","bench","abate","feign","major","death","fresh","crust","stool","colon","abase","marry","react","batty","pride","floss","helix","croak","staff","paper","unfed","whelp","trawl","outdo","adobe","crazy","sower","repay","digit","crate","cluck","spike","mimic","pound","maxim","linen","unmet","flesh","booby","forth","first","stand","belly","ivory","seedy","print","yearn","drain","bribe","stout","panel","crass","flume","offal","agree","error","swirl","argue","bleed","delta","flick","totem","wooer","front","shrub","parry","biome","lapel","start","greet","goner","golem","lusty","loopy","round","audit","lying","gamma","labor","islet","civic","forge","corny","moult","basic","salad","agate","spicy","spray","essay","fjord","spend","kebab","guild","aback","motor","alone","hatch","hyper","thumb","dowry","ought","belch","dutch","pilot","tweed","comet","jaunt","enema","steed","abyss","growl","fling","dozen","boozy","erode","world","gouge","click","briar","great","altar","pulpy","blurt","coast","duchy","groin","fixer","group","rogue","badly","smart","pithy","gaudy","chill","heron","vodka","finer","surer","radio","rouge","perch","retch","wrote","clock","tilde","store","prove","bring","solve","cheat","grime","exult","usher","epoch","triad","break","rhino","viral","conic","masse","sonic","vital","trace","using","peach","champ","baton","brake","pluck","craze","gripe","weary","picky","acute","ferry","aside","tapir","troll","unify","rebus","boost","truss","siege","tiger","banal","slump","crank","gorge","query","drink","favor","abbey","tangy","panic","solar","shire","proxy","point","robot","prick","wince","crimp","knoll","sugar","whack","mount","perky","could","wrung","light","those","moist","shard","pleat","aloft","skill","elder","frame","humor","pause","ulcer","ultra","robin","cynic","agora","aroma","caulk","shake","pupal","dodge","swill","tacit","other","thorn","trove","bloke","vivid","spill","chant","choke","rupee","nasty","mourn","ahead","brine","cloth","hoard","sweet","month","lapse","watch","today","focus","smelt","tease","cater","movie","lynch","saute","allow","renew","their","slosh","purge","chest","depot","epoxy","nymph","found","shall","harry","stove","lowly","snout","trope","fewer","shawl","natal","fibre","comma","foray","scare","stair","black","squad","royal","chunk","mince","slave","shame","cheek","ample","flair","foyer","cargo","oxide","plant","olive","inert","askew","heist","shown","zesty","hasty","trash","fella","larva","forgo","story","hairy","train","homer","badge","midst","canny","fetus","butch","farce","slung","tipsy","metal","yield","delve","being","scour","glass","gamer","scrap","money","hinge","album","vouch","asset","tiara","crept","bayou","atoll","manor","creak","showy","phase","froth","depth","gloom","flood","trait","girth","piety","payer","goose","float","donor","atone","primo","apron","blown","cacao","loser","input","gloat","awful","brink","smite","beady","rusty","retro","droll","gawky","hutch","pinto","gaily","egret","lilac","sever","field","fluff","hydro","flack","agape","wench","voice","stead","stalk","berth","madam","night","bland","liver","wedge","augur","roomy","wacky","flock","angry","bobby","trite","aphid","tryst","midge","power","elope","cinch","motto","stomp","upset","bluff","cramp","quart","coyly","youth","rhyme","buggy","alien","smear","unfit","patty","cling","glean","label","hunky","khaki","poker","gruel","twice","twang","shrug","treat","unlit","waste","merit","woven","octal","needy","clown","widow","irony","ruder","gauze","chief","onset","prize","fungi","charm","gully","inter","whoop","taunt","leery","class","theme","lofty","tibia","booze","alpha","thyme","eclat","doubt","parer","chute","stick","trice","alike","sooth","recap","saint","liege","glory","grate","admit","brisk","soggy","usurp","scald","scorn","leave","twine","sting","bough","marsh","sloth","dandy","vigor","howdy","enjoy","valid","ionic","equal","unset","floor","catch","spade","stein","exist","quirk","denim","grove","spiel","mummy","fault","foggy","flout","carry","sneak","libel","waltz","aptly","piney","inept","aloud","photo","dream","stale","vomit","ombre","fanny","unite","snarl","baker","there","glyph","pooch","hippy","spell","folly","louse","gulch","vault","godly","threw","fleet","grave","inane","shock","crave","spite","valve","skimp","claim","rainy","musty","pique","daddy","quasi","arise","aging","valet","opium","avert","stuck","recut","mulch","genre","plume","rifle","count","incur","total","wrest","mocha","deter","study","lover","safer","rivet","funny","smoke","mound","undue","sedan","pagan","swine","guile","gusty","equip","tough","canoe","chaos","covet","human","udder","lunch","blast","stray","manga","melee","lefty","quick","paste","given","octet","risen","groan","leaky","grind","carve","loose","sadly","spilt","apple","slack","honey","final","sheen","eerie","minty","slick","derby","wharf","spelt","coach","erupt","singe","price","spawn","fairy","jiffy","filmy","stack","chose","sleep","ardor","nanny","niece","woozy","handy","grace","ditto","stank","cream","usual","diode","valor","angle","ninja","muddy","chase","reply","prone","spoil","heart","shade","diner","arson","onion","sleet","dowel","couch","palsy","bowel","smile","evoke","creek","lance","eagle","idiot","siren","built","embed","award","dross","annul","goody","frown","patio","laden","humid","elite","lymph","edify","might","reset","visit","gusto","purse","vapor","crock","write","sunny","loath","chaff","slide","queer","venom","stamp","sorry","still","acorn","aping","pushy","tamer","hater","mania","awoke","brawn","swift","exile","birch","lucky","freer","risky","ghost","plier","lunar","winch","snare","nurse","house","borax","nicer","lurch","exalt","about","savvy","toxin","tunic","pried","inlay","chump","lanky","cress","eater","elude","cycle","kitty","boule","moron","tenet","place","lobby","plush","vigil","index","blink","clung","qualm","croup","clink","juicy","stage","decay","nerve","flier","shaft","crook","clean","china","ridge","vowel","gnome","snuck","icing","spiny","rigor","snail","flown","rabid","prose","thank","poppy","budge","fiber","moldy","dowdy","kneel","track","caddy","quell","dumpy","paler","swore","rebar","scuba","splat","flyer","horny","mason","doing","ozone","amply","molar","ovary","beset","queue","cliff","magic","truce","sport","fritz","edict","twirl","verse","llama","eaten","range","whisk","hovel","rehab","macaw","sigma","spout","verve","sushi","dying","fetid","brain","buddy","thump","scion","candy","chord","basin","march","crowd","arbor","gayly","musky","stain","dally","bless","bravo","stung","title","ruler","kiosk","blond","ennui","layer","fluid","tatty","score","cutie","zebra","barge","matey","bluer","aider","shook","river","privy","betel","frisk","bongo","begun","azure","weave","genie","sound","glove","braid","scope","wryly","rover","assay","ocean","bloom","irate","later","woken","silky","wreck","dwelt","slate","smack","solid","amaze","hazel","wrist","jolly","globe","flint","rouse","civil","vista","relax","cover","alive","beech","jetty","bliss","vocal","often","dolly","eight","joker","since","event","ensue","shunt","diver","poser","worst","sweep","alley","creed","anime","leafy","bosom","dunce","stare","pudgy","waive","choir","stood","spoke","outgo","delay","bilge","ideal","clasp","seize","hotly","laugh","sieve","block","meant","grape","noose","hardy","shied","drawl","daisy","putty","strut","burnt","tulip","crick","idyll","vixen","furor","geeky","cough","naive","shoal","stork","bathe","aunty","check","prime","brass","outer","furry","razor","elect","evict","imply","demur","quota","haven","cavil","swear","crump","dough","gavel","wagon","salon","nudge","harem","pitch","sworn","pupil","excel","stony","cabin","unzip","queen","trout","polyp","earth","storm","until","taper","enter","child","adopt","minor","fatty","husky","brave","filet","slime","glint","tread","steal","regal","guest","every","murky","share","spore","hoist","buxom","inner","otter","dimly","level","sumac","donut","stilt","arena","sheet","scrub","fancy","slimy","pearl","silly","porch","dingo","sepia","amble","shady","bread","friar","reign","dairy","quill","cross","brood","tuber","shear","posit","blank","villa","shank","piggy","freak","which","among","fecal","shell","would","algae","large","rabbi","agony","amuse","bushy","copse","swoon","knife","pouch","ascot","plane","crown","urban","snide","relay","abide","viola","rajah","straw","dilly","crash","amass","third","trick","tutor","woody","blurb","grief","disco","where","sassy","beach","sauna","comic","clued","creep","caste","graze","snuff","frock","gonad","drunk","prong","lurid","steel","halve","buyer","vinyl","utile","smell","adage","worry","tasty","local","trade","finch","ashen","modal","gaunt","clove","enact","adorn","roast","speck","sheik","missy","grunt","snoop","party","touch","mafia","emcee","array","south","vapid","jelly","skulk","angst","tubal","lower","crest","sweat","cyber","adore","tardy","swami","notch","groom","roach","hitch","young","align","ready","frond","strap","puree","realm","venue","swarm","offer","seven","dryer","diary","dryly","drank","acrid","heady","theta","junto","pixie","quoth","bonus","shalt","penne","amend","datum","build","piano","shelf","lodge","suing","rearm","coral","ramen","worth","psalm","infer","overt","mayor","ovoid","glide","usage","poise","randy","chuck","prank","fishy","tooth","ether","drove","idler","swath","stint","while","begat","apply","slang","tarot","radar","credo","aware","canon","shift","timer","bylaw","serum","three","steak","iliac","shirk","blunt","puppy","penal","joist","bunny","shape","beget","wheel","adept","stunt","stole","topaz","chore","fluke","afoot","bloat","bully","dense","caper","sneer","boxer","jumbo","lunge","space","avail","short","slurp","loyal","flirt","pizza","conch","tempo","droop","plate","bible","plunk","afoul","savoy","steep","agile","stake","dwell","knave","beard","arose","motif","smash","broil","glare","shove","baggy","mammy","swamp","along","rugby","wager","quack","squat","snaky","debit","mange","skate","ninth","joust","tramp","spurn","medal","micro","rebel","flank","learn","nadir","maple","comfy","remit","gruff","ester","least","mogul","fetch","cause","oaken","aglow","meaty","gaffe","shyly","racer","prowl","thief","stern","poesy","rocky","tweet","waist","spire","grope","havoc","patsy","truly","forty","deity","uncle","swish","giver","preen","bevel","lemur","draft","slope","annoy","lingo","bleak","ditty","curly","cedar","dirge","grown","horde","drool","shuck","crypt","cumin","stock","gravy","locus","wider","breed","quite","chafe","cache","blimp","deign","fiend","logic","cheap","elide","rigid","false","renal","pence","rowdy","shoot","blaze","envoy","posse","brief","never","abort","mouse","mucky","sulky","fiery","media","trunk","yeast","clear","skunk","scalp","bitty","cider","koala","duvet","segue","creme","super","grill","after","owner","ember","reach","nobly","empty","speed","gipsy","recur","smock","dread","merge","burst","kappa","amity","shaky","hover","carol","snort","synod","faint","haunt","flour","chair","detox","shrew","tense","plied","quark","burly","novel","waxen","stoic","jerky","blitz","beefy","lyric","hussy","towel","quilt","below","bingo","wispy","brash","scone","toast","easel","saucy","value","spice","honor","route","sharp","bawdy","radii","skull","phony","issue","lager","swell","urine","gassy","trial","flora","upper","latch","wight","brick","retry","holly","decal","grass","shack","dogma","mover","defer","sober","optic","crier","vying","nomad","flute","hippo","shark","drier","obese","bugle","tawny","chalk","feast","ruddy","pedal","scarf","cruel","bleat","tidal","slush","semen","windy","dusty","sally","igloo","nerdy","jewel","shone","whale","hymen","abuse","fugue","elbow","crumb","pansy","welsh","syrup","terse","suave","gamut","swung","drake","freed","afire","shirt","grout","oddly","tithe","plaid","dummy","broom","blind","torch","enemy","again","tying","pesky","alter","gazer","noble","ethos","bride","extol","decor","hobby","beast","idiom","utter","these","sixth","alarm","erase","elegy","spunk","piper","scaly","scold","hefty","chick","sooty","canal","whiny","slash","quake","joint","swept","prude","heavy","wield","femme","lasso","maize","shale","screw","spree","smoky","whiff","scent","glade","spent","prism","stoke","riper","orbit","cocoa","guilt","humus","shush","table","smirk","wrong","noisy","alert","shiny","elate","resin","whole","hunch","pixel","polar","hotel","sword","cleat","mango","rumba","puffy","filly","billy","leash","clout","dance","ovate","facet","chili","paint","liner","curio","salty","audio","snake","fable","cloak","navel","spurt","pesto","balmy","flash","unwed","early","churn","weedy","stump","lease","witty","wimpy","spoof","saner","blend","salsa","thick","warty","manic","blare","squib","spoon","probe","crepe","knack","force","debut","order","haste","teeth","agent","widen","icily","slice","ingot","clash","juror","blood","abode","throw","unity","pivot","slept","troop","spare","sewer","parse","morph","cacti","tacky","spool","demon","moody","annex","begin","fuzzy","patch","water","lumpy","admin","omega","limit","tabby","macho","aisle","skiff","basis","plank","verge","botch","crawl","lousy","slain","cubic","raise","wrack","guide","foist","cameo","under","actor","revue","fraud","harpy","scoop","climb","refer","olden","clerk","debar","tally","ethic","cairn","tulle","ghoul","hilly","crude","apart","scale","older","plain","sperm","briny","abbot","rerun","quest","crisp","bound","befit","drawn","suite","itchy","cheer","bagel","guess","broad","axiom","chard","caput","leant","harsh","curse","proud","swing","opine","taste","lupus","gumbo","miner","green","chasm","lipid","topic","armor","brush","crane","mural","abled","habit","bossy","maker","dusky","dizzy","lithe","brook","jazzy","fifty","sense","giant","surly","legal","fatal","flunk","began","prune","small","slant","scoff","torus","ninny","covey","viper","taken","moral","vogue","owing","token","entry","booth","voter","chide","elfin","ebony","neigh","minim","melon","kneed","decoy","voila","ankle","arrow","mushy","tribe","cease","eager","birth","graph","odder","terra","weird","tried","clack","color","rough","weigh","uncut","ladle","strip","craft","minus","dicey","titan","lucid","vicar","dress","ditch","gypsy","pasta","taffy","flame","swoop","aloof","sight","broke","teary","chart","sixty","wordy","sheer","leper","nosey","bulge","savor","clamp","funky","foamy","toxic","brand","plumb","dingy","butte","drill","tripe","bicep","tenor","krill","worse","drama","hyena","think","ratio","cobra","basil","scrum","bused","phone","court","camel","proof","heard","angel","petal","pouty","throb","maybe","fetal","sprig","spine","shout","cadet","macro","dodgy","satyr","rarer","binge","trend","nutty","leapt","amiss","split","myrrh","width","sonar","tower","baron","fever","waver","spark","belie","sloop","expel","smote","baler","above","north","wafer","scant","frill","awash","snack","scowl","frail","drift","limbo","fence","motel","ounce","wreak","revel","talon","prior","knelt","cello","flake","debug","anode","crime","salve","scout","imbue","pinky","stave","vague","chock","fight","video","stone","teach","cleft","frost","prawn","booty","twist","apnea","stiff","plaza","ledge","tweak","board","grant","medic","bacon","cable","brawl","slunk","raspy","forum","drone","women","mucus","boast","toddy","coven","tumor","truer","wrath","stall","steam","axial","purer","daily","trail","niche","mealy","juice","nylon","plump","merry","flail","papal","wheat","berry","cower","erect","brute","leggy","snipe","sinew","skier","penny","jumpy","rally","umbra","scary","modem","gross","avian","greed","satin","tonic","parka","sniff","livid","stark","trump","giddy","reuse","taboo","avoid","quote","devil","liken","gloss","gayer","beret","noise","gland","dealt","sling","rumor","opera","thigh","tonga","flare","wound","white","bulky","etude","horse","circa","paddy","inbox","fizzy","grain","exert","surge","gleam","belle","salvo","crush","fruit","sappy","taker","tract","ovine","spiky","frank","reedy","filth","spasm","heave","mambo","right","clank","trust","lumen","borne","spook","sauce","amber","lathe","carat","corer","dirty","slyly","affix","alloy","taint","sheep","kinky","wooly","mauve","flung","yacht","fried","quail","brunt","grimy","curvy","cagey","rinse","deuce","state","grasp","milky","bison","graft","sandy","baste","flask","hedge","girly","swash","boney","coupe","endow","abhor","welch","blade","tight","geese","miser","mirth","cloud","cabal","leech","close","tenth","pecan","droit","grail","clone","guise","ralph","tango","biddy","smith","mower","payee","serif","drape","fifth","spank","glaze","allot","truck","kayak","virus","testy","tepee","fully","zonal","metro","curry","grand","banjo","axion","bezel","occur","chain","nasal","gooey","filer","brace","allay","pubic","raven","plead","gnash","flaky","munch","dully","eking","thing","slink","hurry","theft","shorn","pygmy","ranch","wring","lemon","shore","mamma","froze","newer","style","moose","antic","drown","vegan","chess","guppy","union","lever","lorry","image","cabby","druid","exact","truth","dopey","spear","cried","chime","crony","stunk","timid","batch","gauge","rotor","crack","curve","latte","witch","bunch","repel","anvil","soapy","meter","broth","madly","dried","scene","known","magma","roost","woman","thong","punch","pasty","downy","knead","whirl","rapid","clang","anger","drive","goofy","email","music","stuff","bleep","rider","mecca","folio","setup","verso","quash","fauna","gummy","happy","newly","fussy","relic","guava","ratty","fudge","femur","chirp","forte","alibi","whine","petty","golly","plait","fleck","felon","gourd","brown","thrum","ficus","stash","decry","wiser","junta","visor","daunt","scree","impel","await","press","whose","turbo","stoop","speak","mangy","eying","inlet","crone","pulse","mossy","staid","hence","pinch","teddy","sully","snore","ripen","snowy","attic","going","leach","mouth","hound","clump","tonal","bigot","peril","piece","blame","haute","spied","undid","intro","basal","shine","gecko","rodeo","guard","steer","loamy","scamp","scram","manly","hello","vaunt","organ","feral","knock","extra","condo","adapt","willy","polka","rayon","skirt","faith","torso","match","mercy","tepid","sleek","riser","twixt","peace","flush","catty","login","eject","roger","rival","untie","refit","aorta","adult","judge","rower","artsy","rural","shave"]
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

@app.post('/digital-colony')
def digital_colony(data: List[DigitalColony] ):
    
    weights = []

    for item in data:
        generations = item.generations
        colony = item.colony
        cd = [0] * 10 
        cp = [[0] * 10 for _ in range(10)] 
        
        digits = [int(d) for d in colony]
        for d in digits:
            cd[d] += 1
        for i in range(len(digits) - 1):
            a, b = digits[i], digits[i + 1]
            cp[a][b] += 1
        
        tw = sum(d * cd[d] for d in range(10))
        
        for generation in range(generations):
            cdn = [0] * 10 
            for a in range(10):
                for b in range(10):
                    c = cp[a][b]
                    if c > 0:
                        diff = abs(a - b)
                        signature = diff if a >= b else 10 - diff
                        new_digit = (tw + signature) % 10
                        cdn[new_digit] += c
            for d in range(10):
                cd[d] += cdn[d]
            tw += sum(d * cdn[d] for d in range(10))
            cpn = [[0] * 10 for _ in range(10)]
            for a in range(10):
                for b in range(10):
                    c = cp[a][b]
                    if c > 0:
                        diff = abs(a - b)
                        signature = diff if a >= b else 10 - diff
                        new_digit = (tw - sum(d * cdn[d] for d in range(10)) + signature) % 10
                        cpn[a][new_digit] += c
                        cpn[new_digit][b] += c
            cp = cpn
        weights.append(str(tw))
    return weights


@app.post("/lisp-parser")
async def interp(data:InterpreterData):
    class String:
        def __init__ (self, content):
            self.content = content
    def format_number(num):
        """Format the number to have a maximum of 4 decimal places without trailing zeros."""
        if isinstance(num, float):
            return float(f"{num:.4f}".rstrip('0').rstrip('.'))
        return num
    def solve(data):
        output = []  
        codes = data.expressions
        symbols = {}  
        print(codes)
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
                    for idx in range(len(args)):
                        if isinstance(args[idx], str):
                            if args[idx] in symbols:
                                args[idx] = symbols[args[idx]]
                            else:
                                error(i)
                    if op == "puts":
                        if size != 1 or not checkType(args[0], String):
                            return error(i)
                        output.append((args[0].content))  # Output the value of the argument (symbol or literal)
                        stack.append(None)
                    elif op == "set":
                        if size != 2 or not isinstance(args[0], str):
                            return error(i)
                        
                        stack.append(None)

                        symbols[args[0]] = (args[1])  # Set a variable in the 'symbols' dictionary
                    elif op == "lowercase":
                        if size != 1 or not checkType(args[0], String):
                            return error(i)
                        stack.append(String((args[0].content).lower())) # Convert the argument to lowercase and push to stack
                    elif op == "uppercase":
                        if size != 1 or not checkType(args[0], String):
                            return error(i)
                        stack.append(String((args[0].content).upper()))  # Convert the argument to uppercase and push to stack
                    elif op == "concat":         
                        if size != 2 or not checkType(args[0], String) or not checkType(args[1], String):
                            return error(i)
                        stack.append(String((args[0].content) + (args[1].content)))
                    elif op == "replace":
                        if size != 3 or not checkType(args[0], String) or not checkType(args[1], String) or not checkType(args[2], String):
                            return error(i)
                        source = (args[0].content)
                        target = (args[1].content)
                        replacement = (args[2].content)
                        result = source.replace(target, replacement)
                        stack.append(String(result)) 
                    elif op == "substring":
                        if size != 3 or not checkType(args[0], String) or not checkType(args[1], int) or not checkType(args[2], int) or args[1] < 0 or args[2] < 0:
                            return error(i) 
                        s = ((args[0]).content)
                        l = ((args[1]))
                        r = ((args[2]))
                        if l < 0 or r < 0 or l >= len(s) or r > len(s) or l > r:
                            return error(i)
                        stack.append(String(s[l:r]))
                    elif op == "add":
                        if size < 2:
                            return error(i)
                        res = 0
                        for num in args:
                            if not checkType(num, (int, float)):  # Check for both int and float
                                return error(i)
                            res += num
                        
                        if isinstance(res, float):
                            res = format_number(res)
                            
                        stack.append(res)

                    elif op == "subtract":
                        if size < 2:
                            return error(i)
                        res = args[0]
                        if not checkType(res, (int, float)):
                            return error(i)
                        for num in args[1:]:
                            if not checkType(num, (int, float)):
                                return error(i)
                            res -= (num)  # Convert to float to handle decimals
                        
                        if isinstance(res, float):
                            res = format_number(res)
                            
                        stack.append(res)

                    elif op == "multiply":
                        if size < 2:
                            return error(i)
                        
                        res = 1
                        for num in args:
                            if not checkType(num, (int, float)):  # Check for both int and float
                                return error(i)
                            res *= (num)  # Convert to float to handle decimals
                        if isinstance(res, float):
                            res = format_number(res)
                        stack.append(res)

                    elif op == "divide":
                        if size != 2 or not checkType(args[0], (int, float)) or not checkType(args[1], (int, float)) or float(args[1]) == 0:
                            return error(i)
                        
                        if isinstance(args[0], int) and isinstance(args[1], int):
                            result = args[0] // args[1]
                        else:
                            result = (args[0]) / (args[1])  # Ensure both arguments are treated as floats
                        stack.append(format_number(result))
                        
                    elif op == "abs":
                        if size != 1 or not checkType(args[0], int):
                            return error(i)
                        stack.append(abs((args[0])))
                    elif op == "max":
                        if size < 2:
                            return error(i)
                        isFloat = False
                        res = (args[0])
                        if isinstance(res, float):
                            isFloat = True
                        if not checkType(res, int):
                            return error(i)
                        for num in args[1::]:
                            if not checkType(num, int):
                                return error(i)
                            if isinstance(num, float):
                                isFloat = True
                            res = max(res, (num))
                            
                        if isFloat:
                            res = format_number(float(res))
                            
                        stack.append(res)
                    elif op == "min":
                        if size < 2:
                            return error(i)
                        res = (args[0])
                        isFloat = False
                        if not checkType(res, int):
                            return error(i)
                        if isinstance(res, float):
                            isFloat = True
                        for num in args[1::]:
                            if not checkType(num, int):
                                return error(i)
                            if isinstance(num, float):
                                isFloat = True
                            res = min(res, (num))
                        
                        if isFloat:
                            res = format_number(float(res))
                            
                        stack.append(res)
                    elif op == "gt":
                        if size != 2 or not checkType(args[0], int) or not checkType(args[1], int):
                            return error(i)
                        stack.append((args[0]) > (args[1]))
                    elif op == "lt":
                        if size != 2 or not checkType(args[0], int) or not checkType(args[1], int):
                            return error(i)
                        stack.append((args[0]) < (args[1]))
                    elif op == "equal":
                        if size != 2:
                            return error(i)
                        stack.append((args[0]) == (args[1]))
                    elif op == "not_equal":
                        if size != 2:
                            return error(i)
                        stack.append((args[0]) != ( args[1]))
                    elif op == "str":
                        if size != 1:
                            return error(i)
                        res = (args[0])
                        if res == None:
                            res = "null"
                        elif isinstance(res, bool):
                            if res:
                                res = "true"
                            else:
                                res = "false"
                        elif isinstance(res, (int, float)):
                            res = str(args[0])
                        else:
                            res = str(res.content)
                        stack.append(String(res))
                    index += 1 



                else:
                    if curr != ' ':

                        if curr == '\"':  # Handle Stringings
                            word = ""
                            index += 1  # Skip the opening quote
                            while index < len(line) and line[index] != '\"':
                                word += line[index]
                                index += 1
                            stack.append(String(word))
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
        print(output)
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
            
# class InputData(BaseModel):
#     guessHistory: List[str]
#     evaluationHistory: List[str]

# @app.post("/wordle-game")
# async def wordle_game(data : InputData):
    
    
    
#     # return {"x": reference_words}
#     five_words_set = set()
    
#     with open('words.txt', 'r') as file:
#         for line in file:
#             # Split each line into words and add them to the set
#             for word in line.split():
#                 five_words_set.add(word)
    
#     # data = {
#     #     "guessHistory": ["slate", "lucky", "maser", "gapes", "wages"], 
#     #     "evaluationHistory": ["?-X-X", "-?---", "-O?O-", "XO-?O", "OOOOO"]
#     # }
    
#     guessHistory = data.guessHistory
#     evaluationHistory = data.evaluationHistory
    
#     if not guessHistory:
#         first_guess = random.choice(list(five_words_set))
#         return {"guess": first_guess}

#     else:
        
#         guess = ['', '', '', '', '']
#         banned_letters = []
#         possible_letters = []
        
#         for i in range(len(evaluationHistory)):
#             prev_guess  = guessHistory[i]
#             prev_eval   = evaluationHistory[i]
            
#             for j in range(len(prev_eval)):
#                 if prev_eval[j] == 'X':
#                     possible_letters.append(prev_guess[j])
#                 elif prev_eval[j] == 'O':
#                     guess[j] = prev_guess[j]
#                     # possible_letters.append(prev_guess[i])
#                 elif prev_eval[j] == '-':
#                     banned_letters.append(prev_guess[j])
#                 elif prev_eval[j] == '?':
#                     continue
        
#         possible_guess = []
#         for word in five_words_set:
            
#             all_possible = True
#             for w in possible_letters:
#                 all_possible = all_possible and w in word
#             if not all_possible:
#                 continue
            
#             put_word = True
#             for i in range(len(word)):
#                 put_word = put_word and not (word[i] in banned_letters)
#                 put_word = put_word and not (guess[i] != '' and guess[i] != word[i])
            
#             if put_word:
#                 possible_guess.append(word)
        
        
#         print(possible_guess)
#         final = possible_guess[0]
        
#         return {
#             "guess": final
#             } 
    
class RequestData(BaseModel):
    dictionary: List[str]
    mistypes: List[str]


@app.post("/the-clumsy-programmer")
async def clumsy(dataList: List[RequestData]):
    res = []
    for data in dataList:
        cur_res = []
        dic, mis = data.dictionary, data.mistypes
        pre = defaultdict(set)
        suf = defaultdict(set)

        for word in dic:
            for i in range(len(word)):
                pre[word[:i]].add(word)
                suf[word[i+1:]].add(word)
        
        for word in mis:
            for i in range(len(word)):
                cur_pre = word[:i]
                cur_suf = word[i+1:]
                if cur_pre in pre and cur_suf in suf:
                    found = False
                    for word in pre[cur_pre]:
                        if word in suf[cur_suf]:
                            cur_res.append(word)
                            found = True
                            break
                    if found:
                        break
        res.append({"corrections": cur_res})

    return res


class OfficeHours(BaseModel):
    timeZone: str
    start: int  # start hour (24-hour format)
    end: int    # end hour (24-hour format)

class User(BaseModel):
    name: str
    officeHours: OfficeHours

class Email(BaseModel):
    subject: str
    sender: str
    receiver: str
    timeSent: str  # ISO 8601 format

class EmailData(BaseModel):
    emails: List[Email]
    users: List[User]

@app.post("/mailtime")
async def mail_time(data: EmailData):
    emails, users = data.emails, data.users
    start_mail = defaultdict(list)
    res = {}
    count = defaultdict(int)
    work_hours = {}
    time_zones = {}

    for user in users:
        work_hours[user.name] = [user.officeHours.start, user.officeHours.end]
        time_zones[user.name] = user.officeHours.timeZone

    for email in emails:
        parts = email.subject.split('RE:')
        # Get the last part and strip whitespace
        subject = parts[-1].strip()
        start_mail[subject].append([len(parts),email.sender, email.receiver, datetime.fromisoformat(email.timeSent)])

    print(start_mail)
    
    def convert_working(dt, start):
        if dt.weekday() >= 5:
            days_until_monday = (7 - dt.weekday()) % 7
            dt += timedelta(days=days_until_monday)
            dt = dt.replace(hour=start, minute=0, second=0, microsecond=0)
            return dt
        
        if not (start <= dt.hour < end):
            while not (start <= dt.hour < end):
                dt += timedelta(hours=1)
            dt = dt.replace(hour=start, minute=0, second=0, microsecond=0)

        if dt.weekday() >= 5:
            days_until_monday = (7 - dt.weekday()) % 7
            dt += timedelta(days=days_until_monday)
            dt = dt.replace(hour=start, minute=0, second=0, microsecond=0)
            return dt
        return dt
    
    def is_working_hour(dt, start, end):
        if dt.weekday() >= 5:  # Saturday and Sunday are considered weekends
            return False
        if start <= dt.hour < end:
            return True
        return False
        
    for subject in start_mail:
        start_mail[subject].sort()
        for i in range(1,len(start_mail[subject])):
            cur, prev = start_mail[subject][i],start_mail[subject][i-1]
            dt1 = cur[-1]
            tz = pytz.timezone(time_zones[cur[1]])
            dt2 = prev[-1].astimezone(tz)
             
            start,end = work_hours[cur[1]]
            time_difference = 0
            dt2 = convert_working(dt2, start)
            dt2_next = dt2.replace(hour=dt2.hour+1, minute=0, second=0, microsecond=0)
            time_difference += dt2_next.timestamp()-dt2.timestamp()
            dt2 = dt2_next

            while dt2 < dt1:
                if is_working_hour(dt2, start, end):
                    time_difference += 60*60
                dt2 += timedelta(hours=1)
          
            time_difference -= dt2.timestamp() - dt1.timestamp()

            res[cur[1]] = res.get(cur[1], 0) + time_difference
            count[cur[1]] += 1
    
    for user in res:
        res[user] /= count[user]
        res[user] = int(res[user])
    
    for user in users:
        if user.name not in res:
            res[user.name] = 0
    
    return {
        "response":res
    }
    
            
            
    
def find_correct_word(mistyped_word: str, dictionary: List[str]) -> str:
    for correct_word in dictionary:
        if len(correct_word) == len(mistyped_word):
            # Count differences
            differences = sum(1 for a, b in zip(correct_word, mistyped_word) if a != b)
            if differences == 1:
                return correct_word
    return None

@app.post("/the-clumsy-programmer")
async def correct_mistypes(dataList: List[RequestData]):
    response = []
    for data in dataList:
        corrections = []
        
        dictionary = data.dictionary
        for mistyped in data.mistypes:
            corrected = find_correct_word(mistyped, dictionary)
            corrections.append(corrected)

        response.append({ "corrections": corrected })
    return response

@app.get("/ub5-flags", response_class=JSONResponse)
async def get_flags():
    # Constructing the expected JSON response
    response_data = {
        "sanityScroll": {
            "flag": "UB5{w3lc0m3_70_c7f_N0ttyB01}"
        },
        "openAiExploration": {
            "flag": "sk-2c90416c24a91a9e1eb18168697e8ff5"
        },
        "dictionaryAttack": {
            "flag": "UB5{FLAG_CONTENT_HERE}",
            "password": "PASSWORD_HERE"
        },
        "pictureSteganography": {
            "flagOne": "UB5-1{FLAG_ONE_CONTENTS_HERE}",
            "flagTwo": "UB5-2{FLAG_TWO_CONTENTS_HERE}"
        },
        "reverseEngineeringTheDeal": {
            "flag": "FLAG_CONTENT_HERE",
            "key": "KEY_HERE"
        }
    }
    
    return response_data
# -------------Wordle Game--------------------



# data1 = {
#    "guessHistory": ["slate", "lucky", "maser"], 
#    "evaluationHistory": ["?-X-X", "-?---", "-O?O-"]
# }
# possible_words = words.copy()
# guess = algorithm(possible_words, history=data1)
# print(guess)

class WordleData(BaseModel):
    guessHistory: List[str]
    evaluationHistory: List[str]

@app.post("/wordle-game")
async def wordle(data : WordleData):
    def validate_guess(history):
        validated_letters = {'yellows': set(), 'greys': set(), 'greens': {}}
        
        guessHistory            = history["guessHistory"]
        evaluationHistory       = history["evaluationHistory"]
        
        for i in range(len(evaluationHistory)):
            # Now, we analyze our past answers and
            # return a validator for future possible guesses
            evalResult = evaluationHistory[i]
            guess      = guessHistory[i]
            # ith guess and result
            for j in range(len(evalResult)):
                # not in the answer at all
                letter = guess[j]
                if evalResult[j] == '-':
                    validated_letters['greys'].add(letter)
                elif evalResult[j] == 'O':
                    validated_letters['greens'][j] = letter
                elif evalResult[j] == 'X':
                    validated_letters['yellows'].add(letter)
                elif evalResult[i] == '?':
                    validated_letters['greens'][j] = letter
                # else case '?' where it is masked
        
        return validated_letters
  
    def narrow_down_words(validated_letters, possible_words):
        filtered_words = []
        with ThreadPoolExecutor(max_workers = 10) as executor:
            for word in possible_words:
                if all(word[i] == letter for i, letter in validated_letters['greens'].items()) and \
                validated_letters['yellows'].issubset(set(word)) and \
                not any(letter in word for letter in validated_letters['greys']):
                    filtered_words.append(word)

        return filtered_words

    # Assuming random guess
    def frequency(possible_words):
        
        guess = ['', '', '', '', '']
        
        for i in range(5):
            freq = defaultdict(int)
            letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
            
            max_freq = 0
            for word in possible_words:
                freq[word[i]] += 1
                max_freq = max(max_freq, freq[word[i]])
            
            for letter, f in freq:
                if f == max_freq:
                    guess[i] = letter
        
        return "".join(guess)
                    
            
            

    def algorithm(possible_words, history=None):
        guess = random.choice(possible_words)

        if not history["evaluationHistory"]:
            return guess
        validated_letters = validate_guess(history)
        possible_words    = narrow_down_words(validated_letters, possible_words)

        
        if not possible_words:
            return guess
        
        guess = frequency(greens(possible_words))
        return guess
    
    history = {}
    history["guessHistory"] = data.guessHistory
    history["evaluationHistory"] = data.evaluationHistory
    
    possible_words = words.copy()
    guess = algorithm(possible_words, history=history)
    
    return {"guess": guess}


@app.post("/coolcodehack")
async def coolcodehack():
    return {
	"username": "asomani",
	"password": "Richie1234."
}
