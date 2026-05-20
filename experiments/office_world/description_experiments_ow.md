## Easy - short policies + fast feedback  (reference expe to show that pql and pqlrm achieve same results)
Name : exp1_time_coffee_mail
Objective : show that pql, pqlrm achieve the same results. Highlight convergence time/steps

Reward sources :
- time_penalty : -1 at each step
- get_mail : returns a reward of 1 when the mail is found and a reward of 1 when the office is reached after obtaining the mail
- get_coffee : returns a reward of 1 when the coffee is found and a reward of 1 when the office is reached after obtaining the coffee

It is possible to remove the reward of 1 obtained when the intermediary objective is reached (coffee or mail reached), but it produces less (interesting) policies and leads to a slower convergence rate

setup : episodes of size 500, gamma = 0.95
epsilon (pql + pqlrm) : 1.0 --> 0.1 in the number of training steps

- pqlrm : 100_000 steps -> 8 policies (3 are incomplete / intermediate policies -> due to the negative reward of time penalty?) / 5 optimal policies : exploration too fast
- pqlrm : 200_000 steps -> 6 optimal policies 
- pql : 500_000 steps -> 6 optimal policies 

## Medium - longer policies than easy setup with less feedback
Name : exp2_mail_coffee_nohit
Objective : show that pqlrm performs better than pql, which has difficulties to find optimal policies

Reward sources : rms no_hit, get_mail, get_coffee
- no_hit : returns a reward of 1 when the agent reaches the office without breaking any decoration (0 otherwise)
- get_mail : returns a reward of 1 when the mail is found and a reward of 1 when the office is reached after obtaining the mail
- get_coffee :  same as get_mail but get coffee

No need of time_penalty since we have gamma < 1, policies produced are the shortest possible.

It is possible to remove the reward of 1 obtained when the intermediary objective is reached, it produces less (interesting) policies

Case of QRM: need to keep the three tasks separate or integrate no hit inside the rm (rms are already designed, just need to change in the setting) // TODO : check because it is so long to run

Same setup as above
setups : episodes of size 500, gamma = 0.95
epsilon : 1.0 --> 0.1 in the number of training steps
epsilon (qrm) : 0.5 (seems to be a good tradeoff, it was 0.1 in the code provided by the authors).

- pqlrm : 100_000 steps -> return the 4 optimal policies
- pql : 500_000 steps -> return 3 policies (not optimal : does not solve the no hit task, increase the number of steps ?)
- qrm : 2_000 episodes (about 335_829 steps) -> all tasks learned

## Hard - longest policies with almost no feedback 
Name : exp3_patrol_nohit
Objective : show that pqlrm finds optimal policies, while pql does not find any policy accomplishing the tasks

2 RMS : no_hit, patrol (with a single reward when all the patrol is done)

Same setup as above
setups : episodes of size 500, gamma = 0.95
epsilon : 1.0 --> 0.1 in the number of training steps
epsilon (qrm) : 0.5

- pqlrm : 100_000 steps -> return the optimal policy
- pql : 500_000 -> no policies learned
- qrm : 2_000 episodes (about 775_915) steps -> tasks learned

