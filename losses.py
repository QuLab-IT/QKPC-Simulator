import numpy as np
import matplotlib.pyplot as plt

from channel.time_dependent_loss   import get_losses
from channel.atmosphere.atmos_data import get_f_atm

def loss_values(loss_params):
    # Get the atmospheric transmission values for the correspondent wv
    f_atm = get_f_atm(loss_params)

    # Get loss data
    loss_data = get_losses( theta_max   = loss_params['theta_max'],
                            loss_params = loss_params,
                            f_atm       = f_atm,
                            tPrint      = False,
                            outpath     = False )

    # Header: Time (s), Elevation (rad), eta_tot, eta_diff, eta_atm, eta_sys, Distance (m)
    time      = loss_data[:,0]
    elev      = loss_data[:,1]
    T_loss    = loss_data[:,2]
    diff_loss = loss_data[:,3]
    atm_loss  = loss_data[:,4]
    int_loss  = loss_data[:,5]
    dist      = loss_data[:,6]

    return {
        "time": time,
        "elev": elev,
        "T_loss": T_loss,
        "diff_loss": diff_loss,
        "atm_loss": atm_loss,
        "int_loss": int_loss,
        "dist": dist,
    }


