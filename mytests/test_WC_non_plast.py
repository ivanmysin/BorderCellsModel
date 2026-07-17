import sys

sys.path.append("..")  # Adds higher directory to python modules path. So you can import parent.pa

from train_wc_nonpsyns import WilsonCowanNetwork





params = {

    "pconn" : 0.0,
    "e_r" : 0.0,
    "tau_pop" : 20,
    'I_ext' : 10.0,
    "gsyn_max" : 1.0,
    "tau_1" : 5.0,
    "tau_2" : 10.0,


}


rnn = WilsonCowanNetwork(params, dt=0.1, batch_size=1)

