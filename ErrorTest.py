
from elephant.spike_train_generation import single_interaction_process as SIP
from quantities import Hz, ms

print SIP(rate=10*Hz, rate_c=1*Hz, n=10, t_stop=100*ms)