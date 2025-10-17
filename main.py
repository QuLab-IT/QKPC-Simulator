"""
Last modified on Fri Oct 17 2025

@author: Lourenço Sumares
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#from scipy.integrate import quad
#from scipy.interpolate import interp1d

import losses
import models


def preprocess_data_ook(file_path):
    
    df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
    df.columns = ['gamma', 'noise', 'Cp', 'col4', 'col5', 'photons']
    
    # Sort for faster lookups
    df = df.sort_values(['gamma', 'noise', 'Cp'], ascending=[True, True, False])
    return df

def preprocess_data_pm(file_path):
    
    df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
    df.columns = ['gamma', 'noise', 'col3', 'col4', 'photons', 'col6', 'Cp']
    
    # Sort for faster lookups
    df = df.sort_values(['gamma', 'noise', 'Cp'], ascending=[True, True, False])
    return df

def find_best_row(df, gamma_value, noise_value):

    if gamma_value > 0.9:
        return gamma_value, 0.0, np.inf
        
    # Exact match first (fast filtering)
    exact_match = df[(df['gamma'] == gamma_value) & (df['noise'] == noise_value)]
    
    if not exact_match.empty:
        return exact_match.iloc[0]['Cp'], exact_match.iloc[0]['photons']
    
    # Calculate minimum euclidean distance
    df['dist'] = (df['gamma'] - gamma_value) ** 2 + (df['noise'] - noise_value) ** 2  
    
    # Find the closest match
    closest_row = df.nsmallest(1, 'dist').iloc[0]
    
    return closest_row['gamma'], closest_row['Cp'], closest_row['photons']



def main():

    parser = argparse.ArgumentParser(description="Simulator of the QKPC protocol")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Simulation mode")

    # --- Mode 1: Standard fixed (rex/gamma) ---
    fixed_parser = subparsers.add_parser("fixed", help="Standard fixed mode: fix rex or gamma")
    fixed_parser.add_argument("model", type=str, choices=["partial", "total"], help="Model (partial/total)")
    fixed_parser.add_argument("variable", type=str, choices=["rex", "gamma"], help="Variable to fix (rex/gamma)")
    fixed_parser.add_argument("value1", type=float, help="Primary value (gamma or rex)")
    fixed_parser.add_argument("modulation", nargs="?", type=str, choices=["ook", "pm"], default="ook", help="Modulation type (ook/pm)")
    fixed_parser.add_argument("--noise", type=float, default=0.002, help="Background noise [photons/pulse]")
    fixed_parser.add_argument("--wavelength", type=float, default=850, help="Wavelength [nm]")
    fixed_parser.add_argument("--aT", type=float, default=0.04, help="Transmitter aperture radius [m]")
    fixed_parser.add_argument("--aR", type=float, default=0.7, help="Receiver aperture radius [m]")
    fixed_parser.add_argument("--aE", type=float, default=0.7, help="Eavesdropper aperture radius [m]")
    fixed_parser.add_argument("--eta_int", type=float, default=15, help="Intrinsic system loss [dB]")
    fixed_parser.add_argument("--more", action="store_true", help="Perform extra calculations")

    # --- Mode 2: Mobile Eve mode (Variable Eve's height) ---
    mobile_parser = subparsers.add_parser("mobile", help="Mobile Eve mode: set number of photons in the receiver")
    mobile_parser.add_argument("value1", type=float, help="Number of photons in the receiver")
    mobile_parser.add_argument("modulation", nargs="?", type=str, choices=["ook", "pm"], default="ook", help="Modulation type (ook/pm)")
    mobile_parser.add_argument("--hE", type=float, default=0, help="Height of the eavesdropper [km]")
    mobile_parser.add_argument("--noise", type=float, default=0.002, help="Background noise [photons/pulse]")
    mobile_parser.add_argument("--wavelength", type=float, default=850, help="Wavelength [nm]")
    mobile_parser.add_argument("--aT", type=float, default=0.04, help="Transmitter aperture radius [m]")
    mobile_parser.add_argument("--aR", type=float, default=0.7, help="Receiver aperture radius [m]")
    mobile_parser.add_argument("--aE", type=float, default=0.7, help="Eavesdropper aperture radius [m]")
    mobile_parser.add_argument("--eta_int", type=float, default=15, help="Intrinsic system loss [dB]")
    mobile_parser.add_argument("--more", action="store_true", help="Perform extra calculations")

    args = parser.parse_args()

    default_values = {
        "wavelength": 850,
        "noise": 0.002,
        "aT": 0.04,
        "aR": 0.7,
        "aE": 0.7,
        "eta_int": 15,
    }

    for arg, default in default_values.items():
        value = getattr(args, arg)
        if value != default:
            print(f"{arg} changed to {value}")

    if args.wavelength <= 785:
        wavelength = 785
    elif args.wavelength >= 850:
        wavelength = 850
    elif args.wavelength > 785 and args.wavelength < 850:
        wavelength = round(args.wavelength / 5) * 5
    
    print(f"Wavelength: {wavelength} nm")

    n = 1.0 # refractive index of the air
    w0 = args.aT / 2

    loss_params = {}

    ######################################################################################################################
    #                                            User Parameters
    ######################################################################################################################

    loss_params['R_E']        = 6371e3              # R_E       = Radius of the Earth [m]
    loss_params['theta_max']  = 90                  # theta_max = Max elevation of satellite orbit wrt receiver [deg]
    loss_params['h_T']        = 500e3               # h_t       = Satellite altitude in [m]
    loss_params['h_R']        = 0                   # hOGS      = Receiver altitude in [m]
    loss_params['aT']         = args.aT             # aT        = Transmitter aperture radii [m]
    loss_params['aR']         = args.aR             # aR        = Receiver aperture radii [m]
    loss_params['w0']         = w0                  # w0        = Beam waist at focus [m]
    loss_params['wvl']        = wavelength          # wl        = Wavelength [nm]
    loss_params['eta_int']    = args.eta_int        # eta_int   = Intrinsic system loss [dB]

    # Ignore the following parameters

    loss_params['tReadLoss']  = False
    loss_params['tWriteLoss'] = False
    loss_params['loss_path']  = ''
    loss_params['loss_file']  = ''
    loss_params['loss_col']   = ''
    loss_params['atm_file']   = ''   

    ######################################################################################################################

    # Convert the efficiency value and theta max value to rads
    loss_params['eta_int']   = 10**(-loss_params['eta_int']/10.0)
    loss_params['theta_max'] = np.deg2rad(loss_params['theta_max'])

    

    wavelength_m = wavelength * 1e-9
    repetition_rate = 1e9 # 1 GHz
    pulse_period = 93e-12 # Delta T 
    big_theta = models.calculate_beam_div(wavelength_m, n, w0)
    print(f"Big theta Divergence: {big_theta} m")
    results = losses.loss_values(loss_params)

    # Preload the dataset
    if args.modulation == "ook":
        df = preprocess_data_ook("filtered_output.txt")
    elif args.modulation == "pm":
        df = preprocess_data_pm("Final_polarization.txt")
    
    if args.mode == "fixed":
        print(f"Running fixed mode with model={args.model}, fixed variable={args.variable}, value={args.value1}")
    
        if args.variable == "rex":

            if args.value1 < 0:

                print(f"Rex must be a positive number!")
            
            elif args.value1 > 0:

                print(f"Fixing Rex with value={args.value1}")
                
                gamma = np.zeros_like(results["time"], dtype=float) 
                approx_gamma = np.zeros_like(results["time"], dtype=float) 
                Cp_max = np.zeros_like(results["time"], dtype=float)  
                photons_value = np.zeros_like(results["time"], dtype=float)

                rex_max = args.value1 * np.sin(math.radians(10))
                #rex_max = args.value1

                if args.model == "partial":

                    for g, t in enumerate(results["time"]):
                        gamma[g] = models.gamma_partial(rex_max, big_theta, results["dist"][g], results["T_loss"][g]/results["diff_loss"][g], args.aE, args.aR)

                elif args.model == "total":

                    for g, t in enumerate(results["time"]):
                        gamma[g] = models.gamma_total(rex_max, big_theta, results["dist"][g], args.aR) # talvez com big_theta/2
                    
                models.plot_gamma_time(results["time"], gamma)
                
                for i, t in enumerate(results["time"]):
                    approx_gamma[i], Cp_max[i], photons_value[i] = find_best_row(df, gamma[i], args.noise)

                # Containers for results
                photons_values = []
                gamma_values = []
                pr_values = []
                updated_time = []
                updated_dist = []
                e_degrees = []
                t_loss_values = []

                # Loop through all values
                for l, f, t, g, p, e, d in zip(results["T_loss"], photons_value, results["time"], approx_gamma, Cp_max, results["elev"], results["dist"]):
                    if  np.degrees(e) > 10:
                        t_loss_values.append(l)
                        photons_values.append(f)
                        gamma_values.append(g)
                        pr_values.append(p)
                        updated_dist.append(d)
                        updated_time.append(t)
                        e_degrees.append(np.degrees(e))

                t_loss_values = np.array(t_loss_values)
                photons_values = np.array(photons_values)    
                gamma_values = np.array(gamma_values)
                pr_values = np.array(pr_values)
                updated_dist = np.array(updated_dist)
                updated_time = np.array(updated_time)

                models.plot_cp(updated_time, pr_values)
                private_rate = pr_values*repetition_rate
                sorted_indices = np.argsort(updated_time)
                updated_time = updated_time[sorted_indices]
                mask = updated_time >= 0
                time_half = updated_time[mask]
                private_rate_half = private_rate[mask]
                private_rate_half = np.maximum(private_rate_half, 0)  # remove negative values
                area_half = np.trapezoid(private_rate_half, time_half)
                total_area = 2 * area_half

                models.plot_photons(updated_time, photons_values)

                print(f"Information sent in a satellite pass: {total_area/1e9} Gbit/pass")

                if args.more:
                    models.plot_as_time(results, photons_values, pulse_period, loss_params, updated_time, t_loss_values)
            

        elif args.variable == "gamma":
            
            rex = np.zeros_like(results["time"], dtype=float) 

            if args.value1 >= 0.9:
                print(f"Fixing gamma with value={args.value1}")
                print(f"Max Private Capacity Cp: 0.0")
                rex[:] = np.inf
                models.plot_rex_time(results["time"], rex)


            elif args.value1 > 0 and args.value1 < 0.9:

                # variables for ellipse
                a = np.zeros_like(results["time"], dtype=float)
                b = np.zeros_like(results["time"], dtype=float) 

                gamma, Cp_max, photons_value = find_best_row(df, args.value1, args.noise)
                print(f"Fixing gamma with value={gamma}")
                print(f"Max Private Capacity Cp: {Cp_max}")
                print(f"Number of Photons at Bob: {photons_value}")
                
                if args.model == "partial":

                    for r, t in enumerate(results["time"]):
                        rex[r] = models.rex_partial(args.value1, big_theta, results["dist"][r], results["T_loss"][r]/results["diff_loss"][r], args.aE, args.aR)

                elif args.model == "total":

                    for r, t in enumerate(results["time"]):
                        rex[r] = models.rex_total(args.value1, big_theta/2, results["dist"][r], args.aR) 

                
                # Containers for results in the updated transmission window
                a_values = []
                b_values = []
                updated_time = []
                updated_dist = []
                e_degrees = []

                # Loop through all values
                for t, r, e, d in zip(results["time"], rex, results["elev"], results["dist"]):
                    if  np.degrees(e) > 10:
                        b = r * (1 / np.sin(e))
                        a_values.append(r)
                        b_values.append(b)
                        updated_dist.append(d)
                        updated_time.append(t)
                        e_degrees.append(np.degrees(e))
                    
                a_values = np.array(a_values)
                b_values = np.array(b_values)
                updated_dist = np.array(updated_dist)
                updated_time = np.array(updated_time)
                total_area = len(updated_time) * Cp_max * repetition_rate

                print(f"Information sent in a satellite pass: {total_area/1e9} Gbit/pass")
                print(f"Minimum rex: {b_values[len(updated_time)//2]} m")     
                print(f"Total losses at zenith: {-10*np.log10(results["T_loss"][len(results["time"])//2])} dB")
                models.plot_bex_time(updated_time, b_values)
                models.plot_aex_time(updated_time, a_values)

                if args.more:
                    models.plot_as_time(results, photons_value, pulse_period, loss_params)
                

            elif args.value1 == 0:
                print(f"Gamma value input must be greater than 0")

    elif args.mode == "mobile":
        print(f"Running mobile Eve mode with {args.value1} photons at the receiver when Eve is at {args.hE} km.")
        alpha_squared = args.value1
        loss_params["h_R"] = args.hE * 1e3
        gamma = 0.1
        models.plot_mobile_eve(loss_params, alpha_squared, gamma, pulse_period, args.more)


    else:
        parser.error("Unknown mode selected")

 
if __name__ == '__main__':
    main()

