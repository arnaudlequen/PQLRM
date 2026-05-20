## Comparison of QRM, PQL, PQLRM
Name : exp1_time_treasure_pressure

Reward sources :
- time_penalty : -1 at each step
- build_pbst_rm_treasure : returns a reward depending on the treasure reached
- build_pbst_rm_pressure_v2 : returns a penalty depending on the number of consecutive down actions performed while reaching a treasure

setup : episodes of size 300, gamma = 0.999
epsilon (pql + pqlrm) : 1.0 --> 0.1 in the number of training steps
epsilon (qrm) : 0.5

- pqlrm : 100_000 steps -> all the 15 optimal policies
- pql : 200_000 steps -> all the 15 optimal policies
- qrm : 10_000 episodes -> tasks time_penalty and pressure very easy, for treasure it does not find the deepest ones



