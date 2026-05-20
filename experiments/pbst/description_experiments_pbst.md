## Comparison of QRM, PQL, PQLRM
Name : exp1_time_treasure_pressure

Reward sources :
- time_penalty : -1 at each step
- build_pbst_rm_treasure : returns a reward depending on the treasure reached
- build_pbst_rm_pressure_v2 : returns a penalty depending on the number of consecutive down actions performed while reaching a treasure

setup : episodes of size 100, gamma = 0.95
epsilon (pql + pqlrm) : 1.0 --> 0.1 in the number of training steps
epsilon (qrm) : 0.5

- pqlrm : 100_000 steps -> 8 policies (3 are incomplete / intermediate policies -> due to the negative reward of time penalty?) / 5 optimal policies : exploration too fast
- pqlrm : 200_000 steps -> 6 optimal policies 
- pql : 500_000 steps -> 6 optimal policies 

