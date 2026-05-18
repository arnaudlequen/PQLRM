## (Medium, intermediate feedback) exp1_mail_coffee_nohit
3 RMS : no_hit, get_mail, get_coffee
- no_hit : returns a reward of 1 when the agent reaches the office without breaking any decoration (0 otherwise)
- get_mail : returns a reward of 1 when the mail is found and a reward of 1 when the office is reached after obtaining the mail
- get_coffee :  same as get_mail but get coffee

No need of time_penalty since we have gamma < 1

Case of QRM: need to keep the three tasks separate or integrate no hit inside the rm (rms are already designed, just need to change in the setting) // TODO : check because it is so long to run

It is possible to remove the reward of 1 obtained when the intermediary objective is reached, it produces less (interesting) policies

setups : episodes of size 500, gamma = 0.95
epsilon : 1.0 --> 0.1 in the number of training steps

- pqlrm : 100000 steps -> return the 4 optimal policies
- pql : 500000 steps -> return 3 policies (not optimal : does not solve the no hit task, increase the number of steps ?)
- qrm : ???

## (Hard, almost no feedback) exp2_patrol_nohit
2 RMS : no_hit, patrol (with a single reward when all the patrol is done)

Same setup as above

- pqlrm : 100_000 steps -> return the optimal policy
- pql : 500_000 -> no policies learned
- qrm : ?

## (Easy, short policies and feedback) exp3_time_coffee_mail (reference expe to show that pql and pqlrm achieve same results)

Same setup

- pqlrm : 100_000 steps -> 8 policies (3 are incomplete / intermediate policies -> due to the negative reward of time penalty?) / 5 optimal policies : exploration too fast
- pqlrm : 200_000 steps -> 6 optimal policies 
- pql : 500_000 steps -> 6 optimal policies 
- qrm : ?