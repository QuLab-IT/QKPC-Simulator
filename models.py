import numpy as np
import matplotlib.pyplot as plt
import losses as lss
import private_capacity as pc
from matplotlib import colormaps
from matplotlib.cm import rainbow
from scipy.stats import skellam
from scipy.special import factorial
from scipy.constants import h
from scipy.constants import c


def calculate_power(n_photons, wavelength, pulse_period):

    photon_energy = h * c / wavelength
    power = n_photons * photon_energy / pulse_period
    return power

def calculate_n_photons(power, wavelength, pulse_period):

    photon_energy = h * c / wavelength
    n_photons = power * pulse_period / photon_energy
    return n_photons

# considering variable losses
def calculate_alice_power(L, bob_photons, wavelength, pulse_period, w0, aR):

    power_ground = calculate_power(bob_photons, wavelength, pulse_period)  
    alice_power = power_ground / L
    return alice_power


###### STATIC EVE ###### 
def rex_partial(gamma, beam_div, d, eta_b, r_eve, r_bob):

    return 0.5* beam_div * d * np.sqrt(0.5 * np.log(1/gamma * 1/eta_b * (r_eve/r_bob)**2))

def rex_total(gamma, beam_div, d, r_bob):

    return beam_div * d * np.sqrt(-0.5 * np.log(gamma *(1 - np.exp(-2*(r_bob/(beam_div * d)) ** 2))))


def gamma_partial(rex, beam_div, d, eta_b, r_eve, r_bob):

    numerator = np.exp(-2*(2*rex/(beam_div * d)) ** 2) * (r_eve**2)
    denominator = eta_b * (r_bob**2)

    return numerator/denominator

def gamma_total(rex, beam_div, d, r_bob):

    numerator = np.exp(-2*(rex/(beam_div * d)) ** 2)
    denominator = 1 - np.exp(-2*(r_bob/(beam_div * d)) ** 2)

    return numerator/denominator



def beam_radius(d, big_theta):

    return d * np.tan(big_theta / 2)


def calculate_beam_div(wavelength, n, omega_zero):

    # omega_zero is the beam waist
    small_theta = wavelength / (np.pi * n * omega_zero)
    big_theta = 2 * small_theta

    return big_theta


def plot_as_time(results, photons_value, pulse_period, loss_params, updated_time, t_loss_updated, save=False):

    wavelength = loss_params['wvl'] * 1e-9
    Pt = calculate_alice_power(t_loss_updated, photons_value, wavelength, pulse_period, loss_params['w0'], loss_params['aR'])
    np.savetxt("T_loss_updated.txt", t_loss_updated)
    np.savetxt("photons_value.txt", photons_value)

    # --- Plot 1: Alice Power ---
    plt.figure(figsize=(7, 4.5))
    plt.plot(updated_time, 10*np.log10(Pt), color="black", linewidth=1.2)
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Alice Power [dB]", fontsize=16)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    if save: plt.savefig("alice_power.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Plot 2: Photon counts ---
    photons_t = calculate_n_photons(Pt, wavelength, pulse_period)
    plt.figure(figsize=(7, 4.5))
    plt.plot(updated_time, photons_t, color="black", linewidth=1.2)
    plt.yscale("log")
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Number of Photons", fontsize=16)
    plt.grid(True, linestyle=":", alpha=0.6, which="both")
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    if save: plt.savefig("photon_counts.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Plot 3: Total channel loss ---
    loss_dB = -10*np.log10(results["T_loss"])
    plt.figure(figsize=(7, 4.5))
    plt.plot(results["time"], loss_dB, color="black", linewidth=1.2)

    # Find minimum value
    min_idx = np.argmin(loss_dB)
    min_time = results["time"][min_idx]
    min_val = loss_dB[min_idx]

    # Add annotation (balloon/arrow)
    plt.annotate(f"Min: {min_val:.2f} dB", xy=(min_time, min_val), xytext=(min_time, min_val + 15), ha = 'center', va='bottom', arrowprops=dict(facecolor="black", shrink=4, width=1, headwidth=5), fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Total Loss [dB]", fontsize=16)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    if save: plt.savefig("total_loss_with_min.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Plot 4: Loss components ---
    plt.figure(figsize=(7, 4.5))
    #plt.plot(np.degrees(results["elev"]), -10*np.log10(results["diff_loss"]), label="Geometric", linewidth=1.2, marker="o", markersize=4, markevery=15)
    #plt.plot(np.degrees(results["elev"]), -10*np.log10(results["atm_loss"]), label="Atmospheric", linewidth=1.2, marker="s", markersize=4, markevery=15)
    #plt.plot(np.degrees(results["elev"]), -10*np.log10(results["int_loss"]), label="Intrinsic", linewidth=1.2, marker="^", markersize=4, markevery=15)
    #plt.xlabel("Elevation [°]", fontsize=16)
    plt.plot(results["time"], -10*np.log10(results["diff_loss"]), label="Geometric", linewidth=1.2, marker="o", markersize=4, markevery=15)
    plt.plot(results["time"], -10*np.log10(results["atm_loss"]), label="Atmospheric", linewidth=1.2, marker="s", markersize=4, markevery=15)
    plt.plot(results["time"], -10*np.log10(results["int_loss"]), label="Intrinsic", linewidth=1.2, marker="^", markersize=4, markevery=15)
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Loss [dB]", fontsize=16)
    plt.legend(fontsize=14, frameon=False)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    if save: plt.savefig("loss_components.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_photons(time, photons):

    plt.plot(time, photons, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('photons')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def plot_dist(time, dist):

    plt.plot(time, dist, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('Distance ')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def total_losses(time, T_loss):

    plt.plot(time, -10*np.log10(T_loss), color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('Total Loss [dB]')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def plot_gamma_time(time, gamma):

    plt.plot(time, gamma, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('$\\gamma$')
    plt.ylim(-0.1, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def plot_cp(time, cp):

    plt.plot(time, cp, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('$C_{P} $')
    plt.ylim(-0.1, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def plot_rate(time, rate):

    plt.plot(time, rate, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('Private rate [Hz]')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def plot_rex_time(time, rex):

    plt.plot(time, rex, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('Exclusion radius - $r_{ex}$ [m]')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def plot_bex_time(time, rex):

    plt.plot(time, rex, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('Exclusion radius - $b_{ex}$ [m]')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()

def plot_aex_time(time, rex):

    plt.plot(time, rex, color="black", linewidth=1.5)
    plt.xlabel('time [s]')
    plt.ylabel('Exclusion radius - $a_{ex}$ [m]')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.show()



def plot_rex_d(gamma_values, beam_div, r_bob, wavelength, n, omega_zero, r_eve, eta_bob):

    # Initialize the plot
    plt.figure(figsize=(10, 10))

    d = np.arange(0.5, 1.3e6, 2e5)

    # Colors for each gamma value
    colors = ['blue', 'red']

    # Loop through the provided gamma values (0.1 and 0.9)
    for i, gamma in enumerate(gamma_values):
        # Calculate the exclusion radius for the current gamma
        rex_total = rex_total(gamma, beam_div, d, r_bob)
        rex_partial = rex_partial(gamma, beam_div, d, eta_bob, r_eve, r_bob)

        # Plot the exclusion radius as a function of d
        plt.plot(d, rex_total, label=f'Total, $\\gamma=${gamma}', linestyle='-', color=colors[i], linewidth=4)
        # Plot the exclusion radius as a function of d
        plt.plot(d, rex_partial, label=f'Partial, $\\gamma=${gamma}', linestyle='--', color=colors[i], linewidth=4)


    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('$d ~[m]$', fontsize=32)
    plt.ylabel('$r_{ex} ~[m]$', fontsize=32)

    # Define y-ticks every 100 units, from 0 to 1000
    plt.yticks(np.arange(0, 1100, 50))

    plt.legend(fontsize=16, loc='upper right')
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylim(0, 200)
    plt.xlim(0, 1.2e6)

    # Enable scientific notation on x-axis
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # Increase font size of scientific notation (offset text)
    plt.gca().xaxis.get_offset_text().set_fontsize(30)

    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.show()


###### Mobile EVE ######
def loss_h(loss_params, h_R):

    loss_params['h_R'] = h_R
    if h_R == 0:
        results = lss.loss_values(loss_params) 
        return results["time"], results["elev"], results["T_loss"], results["dist"] #MUDAR PARA T_LOSS
    elif  0 < h_R <= 20e3:
        results = lss.loss_values(loss_params)
        tot_losses = results["diff_loss"] * results["atm_loss"]
        return results["time"], results["elev"], tot_losses, results["dist"]
    elif 20e3 < h_R <= 500e3:
        results = lss.loss_values(loss_params)
        return results["time"], results["elev"], results["diff_loss"], results["dist"]

def calculate_power_satellite(alpha_squared, wavelength, pulse_period, loss_params, h_R):
    """
    Calculates the required power at the satellite and ground to capture |alpha|^2 photons.
    Returns:
    - Dictionary with calculated values.
    """
    # Power needed at the ground to generate |alpha|^2 photons per pulse
    power_ground = calculate_power(alpha_squared, wavelength, pulse_period)
    time, elev, attenuation, dist = loss_h(loss_params, h_R)

    z = dist  # Distance to the ground (m)
    power_satellite = power_ground * (1/attenuation)

    return {
        "power_ground": power_ground,
        "power_satellite": power_satellite,
        "orbital_distance": dist,
        "elev": elev,
        "eta_diff": attenuation
    }

def beam_power_at_distance_orbit(power_satellite, distance, loss_params):
    """
    Calculates the beam power at a point along the link, assuming that Eve moves along the orbit with the same distance.

    Parameters:
    - power_satellite: initial power at the satellite (in watts).
    - distance: distance from the point where the power will be calculated (in meters).
    - total_distance: total link distance (in meters).

    Returns:
    - Beam power at the specified points (in watts).
    """
    h_R = loss_params['h_T'] - distance
    _, _, attenuation, _ = loss_h(loss_params, h_R)
    loss = attenuation[len(attenuation)//2]
    beam_power_t = power_satellite * loss
    return beam_power_t


def beam_radius_for_gamma(alpha_squared, gamma, wavelength, w0, pulse_period, distances, power_satellite, loss_params):
    """
    Calculates the beam radius at each point along the link to ensure that a fraction gamma*|alpha|^2
    photons are outside the radius.

    Returns:
    - Array with the beam radii at each point along the link.
    """
    h = 6.626e-34  # Planck's constant (J·s)
    c = 3e8         # Speed of light (m/s)

    # Light frequency
    frequency = c / wavelength

    # Calculate the beam radius at each point
    radii = []
    w_z_array = []
    powers_at_point = []
    if np.isscalar(distances):
        z = distances
        power_at_point = beam_power_at_distance_orbit(power_satellite, z, loss_params)
        powers_outside = gamma * alpha_squared * h * frequency / pulse_period
        ratio = powers_outside / power_at_point
        #ratio = np.full_like(power_satellite, gamma)
        w_z = w0 * np.sqrt(1 + (wavelength * z / (np.pi * w0**2))**2)  # Beam radius
        r_outside = np.sqrt(-0.5 * w_z**2 * np.log(ratio))  # Corrected radius for outside power
        return r_outside, w_z, power_at_point
    else:
        for z in distances:
            # Calculate power at the current distance
            power_at_point = beam_power_at_distance_orbit(power_satellite, z, loss_params)
            powers_outside = gamma * alpha_squared * h * frequency / pulse_period
            ratio = powers_outside / power_at_point
            #ratio = gamma
            w_z = w0 * np.sqrt(1 + (wavelength * z / (np.pi * w0**2))**2)  # Beam radius
            r_outside = np.sqrt(-0.5 * w_z**2 * np.log(ratio))  # Corrected radius for outside power
            
            #print(f"Distance: {z:.3e}, Beam Radius: {w_z:.3e}, Radius for photons outside: {r_outside:.3e}, Power outside: {powers_outside:.3e} W, Ratio: {ratio:.3e}")
            powers_at_point.append(power_at_point)
            w_z_array.append(w_z)
            radii.append(r_outside)

        return np.array(radii), np.array(w_z_array), np.array(powers_at_point)
    

# plot the diffraction losses or the total losses for 20 different receiver altitudes, as a function of the time. Vagina type graph for diff. Deer for total.
def plot_hR_time(loss_params, h_R):
    
    plt.figure(figsize=(12, 7))
    loss_params['h_R'] = h_R
    time, elev, T_loss, eta_diff, atm_loss, int_loss, dist = lss.loss_values(loss_params)

    plt.plot(time, -10*np.log10(T_loss), label=f"$h_E$ = {h_R/1e3:.0f} km", color='black', lw=1.8)

    # Plot formatting
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Total Loss [dB]', fontsize=16)
    #plt.title('Time-varying Loss (T_loss) at Different Receiver Altitudes', fontsize=16)
    plt.legend(loc='upper right', fontsize=16, ncol=1)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_mobile_eve(loss_params, alpha_squared, gamma, pulse_period, flag):

    distance = 500e3 - loss_params["h_R"]
    wavelength = loss_params['wvl'] * 1e-9
    results = calculate_power_satellite(alpha_squared, wavelength, pulse_period, loss_params, 0) 
    
    # Compute exclusion radius, beam radius and power for the current Eve's height
    radii, w_z_array, power_val = beam_radius_for_gamma(alpha_squared, gamma, wavelength, loss_params['w0'], pulse_period, distance, results["power_satellite"], loss_params)
    stretched_rex = []
    updated_dist = []
    for e, r, d in zip(results["elev"], radii, results["orbital_distance"]):
        if  np.degrees(e) > 10:
            srex = r * (1 / np.sin(e))
            stretched_rex.append(srex)
            updated_dist.append(d)

    stretched_rex = np.array(stretched_rex)
    updated_dist = np.array(updated_dist)
    plt.figure(figsize=(12, 7))
    label_1 = f"$a_{{ex}}$, $h_E$ = {loss_params["h_R"]/1e3:.0f} km"
    label_2 = f"$b_{{ex}}$, $h_E$ = {loss_params["h_R"]/1e3:.0f} km"
    plt.plot(results["orbital_distance"] / 1e3, radii, label=label_1, color='blue')
    plt.plot(updated_dist / 1e3, stretched_rex, label=label_2, color='red')
    plt.xlim(500, updated_dist[len(updated_dist)-1]/1e3)
    plt.xlabel("Orbital distance [km]", fontsize = 16)
    plt.ylabel("Exclusion radius [m]", fontsize=16)
    #plt.title("Exclusion radius vs Orbital Distance for Different Receiver Distances")
    plt.grid(True)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    plt.show() 

    plt.figure(figsize=(12, 7))
    label = f"$h_E$ = {loss_params["h_R"]/1e3:.0f} km"
    plt.plot(results["orbital_distance"] / 1e3, 10*np.log10(power_val), label=label, color = 'blue')
    plt.xlabel("Orbital distance [km]", fontsize=16)
    plt.ylabel("Power at point [dBW]", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    plt.show()

    if flag:
        plt.figure(figsize=(12, 7))
        plt.plot(results["orbital_distance"] / 1e3, 10*np.log10(results["power_satellite"]), label=label, color = 'black')
        plt.xlabel("Orbital distance [km]", fontsize=16)
        plt.ylabel("Power at satellite [dBW]", fontsize=16)
        plt.grid(True)
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tick_params(axis="both", labelsize=16)
        plt.tight_layout()
        plt.show()

        #plot_hR_time(loss_params, loss_params["h_R"])