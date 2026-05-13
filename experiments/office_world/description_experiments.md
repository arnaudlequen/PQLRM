## exp1_mail_coffee_nohit
3 RMS : no_hit, get_mail, get_coffee
- no_hit : returns a reward of 1 when the agent reaches the office without breaking any decoration (0 otherwise)
- get_mail : returns a reward of 1 when the mail is found and a reward of 1 when the office is reached after obtaining the mail
- get_coffee :  same as get_mail but get coffee

It is possible to remove the reward of 1 obtained when the intermediary objective is reached, it produces less (interesting) policies

setups : episodes of size 500, gamma = 0.95, 
- pqlrm : 100000 steps -> return the 4 optimal policies
- pql : 400000 steps -> return 3 policies (not optimal, increase the number of steps ?)
