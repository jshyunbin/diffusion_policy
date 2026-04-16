# TODO list
- The CVAE encoder in ACT and GRAM implementation is somewhat wrong
    - needs (\[CLS\] + action sequence tokens + \[joint token\]) to 
    be passed through transformer and aggregated.
    - Take the output of \[CLS\] token and do a linear pass to 
    generate mu and sigma -> sample from a normal distribution to 
    generate z

- GRAM needs Q-head for early halting option and V-head to reward 
better action generating models when running parallel