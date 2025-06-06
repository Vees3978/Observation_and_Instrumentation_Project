# -*- coding: utf-8 -*-
"""finalcodestuff.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N1D8gyQmhtoRZfSc6Gn5vT2c9oLPGemP
"""

def convert_phots_to_watts(photons_per_pixel, telescope):
    """
    Convert flux from photons/pixel to watts/m^2.
    photons_per_pixel: Number of photons per pixel.
    return: Flux in watts per square meter.
    """
    if telescope == Artemis:
        pixel_area = (15.2 * 1e-6)**2 #m^2
        wavelength = (550 * 1e-9) #green

    if telescope == Apollo:
        pixel_area = (15.2 * 1e-6)**2 #m^2
        wavelength = (650 * 1e-9)

    if telescope == Leto:
        pixel_area = (15.2 * 1e-6)**2 #m^2
        wavelength = (775 * 1e-9)

    time_interval=2

    h = 6.62607015e-34  # Plancks: J*s
    c = 3e8             # m/s

    frequency = c / wavelength

    energy_per_photon = h * frequency

    total_energy = energy_per_photon * photons_per_pixel

    power = total_energy / time_interval

    flux_watts_per_m2 = power / pixel_area

    return flux_watts_per_m2

import numpy as np

def flux_to_magnitude(flux, flux_error, zero_point=0):
    """
    Convert flux values to magnitudes.

    Parameters:
    - flux (array): Array of flux values in W/m^2.
    - flux_error (array): Array of flux errors.
    - zero_point (float): Zero point constant for the conversion. Default is 0 for relative magnitudes.

    Returns:
    - magnitudes (array): Array of magnitudes.
    - magnitude_errors (array): Array of magnitude errors.
    """
    magnitudes = -2.5 * np.log10(flux) + zero_point
    magnitude_errors = 2.5 / np.log(10) * (flux_error / flux)

    return magnitudes, magnitude_errors

magnitudes_ap = -2.5 * np.log10(1.3)
print(magnitudes_ap)

np.max(flux[s_time:e_time])

flux = # whatever the flux array is from the table

flare_candidate = np.zeros_like(u_image, dtype=bool)

# create a mask for exposures that hare 3sigma above only if the one before or after is also 3sigma is above
for i in range(len(u_image)):
  three_sigma = np.average(u_image[i]) * 3 # used the avg pixel uncertainty of each exposure
  if flux[i] > three_sigma:
        if (i > 0 and flux[i-1] > three_sigma) or (i < len(u_image)-1 and flux[i+1] > three_sigma):
            flare_candidate[i] = True

# should be able to mask now :)
flux[flare_candidate]

# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd


import scipy.stats as stats
import scipy.signal as sig

#extracting from txt files

apollo_exp_indicies = []
apollo_exp_times_epoch = []
apollo_target_offsets = []
apollo_ref_offsets = []
apollo_target_flux = []
apollo_ref_flux = []
apollo_snr = []

#1702014814.786133

f = open("apollo_r_data.txt", "r")
for i, line in enumerate(f):
    if i == 0:
        continue
    data = line.split(",")

    apollo_exp_indicies.append(int(data[0]))
    apollo_exp_times_epoch.append(float(data[1]))
    apollo_target_offsets.append([float(data[2]), float(data[3])])
    apollo_ref_offsets.append([float(data[4]), float(data[5])])
    apollo_target_flux.append(float(data[6]))
    apollo_ref_flux.append(float(data[7]))
    apollo_snr.append(float(data[8]))

artemis_exp_indicies = []
artemis_exp_times_epoch = []
artemis_target_offsets = []
artemis_ref_offsets = []
artemis_target_flux = []
artemis_ref_flux = []
artemis_snr = []

#1702014814.786133
#80 pixels diameter for all apetures.
f = open("artemis_g_data.txt", "r")
for i, line in enumerate(f):
    if i == 0:
        continue
    data = line.split(",")

    artemis_exp_indicies.append(int(data[0]))
    artemis_exp_times_epoch.append(float(data[1])) #epoch
    artemis_target_offsets.append([float(data[2]), float(data[3])])
    artemis_ref_offsets.append([float(data[4]), float(data[5])])
    artemis_target_flux.append(float(data[6]))  # ADU/Pixel**2
    artemis_ref_flux.append(float(data[7]))
    artemis_snr.append(float(data[8]))

leto_exp_indicies = []
leto_exp_times_epoch = []
leto_target_offsets = []
leto_ref_offsets = []
leto_target_flux = []
leto_ref_flux = []
leto_snr = []

#1702014814.786133

f = open("leto_i_data.txt", "r")
for i, line in enumerate(f):
    if i == 0:
        continue
    data = line.split(",")

    leto_exp_indicies.append(int(data[0]))
    leto_exp_times_epoch.append(float(data[1]))
    leto_target_offsets.append([float(data[2]), float(data[3])])
    leto_ref_offsets.append([float(data[4]), float(data[5])])
    leto_target_flux.append(float(data[6]))
    leto_ref_flux.append(float(data[7]))
    leto_snr.append(float(data[8]))

#Note: Leto is off by three seconds so there is about one itteration that will probably be off about leto

artemis_current_time_utc = []

# Convert epoch to UTC
for epoch_timestamp in artemis_exp_times_epoch:
    utc_time = datetime.utcfromtimestamp(epoch_timestamp)
    artemis_current_time_utc.append(utc_time)

apollo_current_time_utc = []

# Convert epoch to UTC
for epoch_timestamp in apollo_exp_times_epoch:
    utc_time = datetime.utcfromtimestamp(epoch_timestamp)
    apollo_current_time_utc.append(utc_time)

leto_current_time_utc = []

# Convert epoch to UTC
for epoch_timestamp in leto_exp_times_epoch:
    utc_time = datetime.utcfromtimestamp(epoch_timestamp)
    leto_current_time_utc.append(utc_time)

# converting to seconds

apo_time_differences_seconds = [(time - apollo_current_time_utc[0]).total_seconds() for time in apollo_current_time_utc]

apo_time_differences_seconds = np.array(apo_time_differences_seconds)


art_time_differences_seconds = [(time - artemis_current_time_utc[0]).total_seconds() for time in artemis_current_time_utc]

art_time_differences_seconds = np.array(art_time_differences_seconds)

leto_time_differences_seconds = [(time - leto_current_time_utc[0]).total_seconds() for time in leto_current_time_utc]

leto_time_differences_seconds = np.array(leto_time_differences_seconds)

def convert_phots_to_watts(photons_per_pixel, telescope):
    """
    Convert flux from photons/pixel to watts/m^2.
    photons_per_pixel: Number of photons per pixel.
    return: Flux in watts per square meter.
    """
    if telescope == 'Artemis':
        pixel_area = (15.2 * 1e-6)**2 #m^2
        wavelength = (550 * 1e-9) #green

    if telescope == 'Apollo':
        pixel_area = (15.2 * 1e-6)**2 #m^2
        wavelength = (650 * 1e-9)

    if telescope == 'Leto':
        pixel_area = (15.2 * 1e-6)**2 #m^2
        wavelength = (775 * 1e-9)

    time_interval=2

    h = 6.62607015e-34  # Plancks: J*s
    c = 3e8             # m/s

    frequency = c / wavelength

    energy_per_photon = h * frequency

    total_energy = energy_per_photon * photons_per_pixel

    power = total_energy / time_interval

    flux_watts_per_m2 = power / pixel_area

    return flux_watts_per_m2

# converting the flux

apo_norm_flux = convert_phots_to_watts((np.array(apollo_target_flux)/ np.array(apollo_ref_flux))*np.median(np.array(apollo_target_flux)/ np.array(apollo_ref_flux)), 'Apollo')

art_norm_flux = convert_phots_to_watts((np.array(artemis_target_flux)/ np.array(artemis_ref_flux))*np.median(np.array(artemis_target_flux)/ np.array(artemis_ref_flux)),'Artemis')

let_norm_flux = (np.array(leto_target_flux)/ np.array(leto_ref_flux))*np.median(np.array(leto_target_flux)/ np.array(leto_ref_flux))

# Calculate subplots
time_range = 500
num_subplots = int(max(max(apo_time_differences_seconds), max(art_time_differences_seconds)) / time_range) + 1
num_columns = 4

# Calculate the number of rows needed
num_rows = -(-num_subplots // num_columns)

# Set the size of each subplot
subplot_width = 6
subplot_height = 4


fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * subplot_width, num_rows * subplot_height), sharey=True)

# Iterate through each subplot
for i in range(num_rows):
    for j in range(num_columns):
        subplot_index = i * num_columns + j
        if subplot_index < num_subplots:
            # Define the time range for the current subplot
            start_time = subplot_index * time_range
            end_time = (subplot_index + 1) * time_range

            # Filter data within the current time range
            apo_mask = (apo_time_differences_seconds >= start_time) & (apo_time_differences_seconds < end_time)
            art_mask = (art_time_differences_seconds >= start_time) & (art_time_differences_seconds < end_time)

            # Find the minimum and maximum time values for each dataset within the specified time range
            apo_min_time = np.min(np.extract(apo_mask, apo_time_differences_seconds))
            apo_max_time = np.max(np.extract(apo_mask, apo_time_differences_seconds))
            art_min_time = np.min(np.extract(art_mask, art_time_differences_seconds))
            art_max_time = np.max(np.extract(art_mask, art_time_differences_seconds))

            # Plot the fluxes for the current subplot with increased linewidth
            ax = axs[i, j]
            ax.plot(apo_time_differences_seconds[apo_mask], (apo_norm_flux[apo_mask]*1e8), label='Apollos Flux in r', linewidth=2, color = 'black')
            ax.plot(art_time_differences_seconds[art_mask], art_norm_flux[art_mask]*1e8, label='Artemis Flux in g', linewidth=2, color='red')

            # Set labels and title for the subplot
            ax.set_xlabel('Time (s)', fontsize = 17)
            ax.set_ylabel('Flux ($\\frac{W}{m^2}$)',fontsize = 17)
            ax.set_title(f'Time Range: {start_time} to {end_time} seconds',fontsize = 20)

            # Set x-axis limits to the minimum and maximum time values for each dataset within the specified time range
            ax.set_xlim(min(apo_min_time, art_min_time), max(apo_max_time, art_max_time))

# Add a single legend for the entire plot
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',fontsize= 20,bbox_to_anchor=(.5, -0.05))

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to make space for the common legend

# Save the plot to a file (e.g., PNG, PDF, SVG)
#plt.savefig('/Users/veronicaestrada/Downloads/UnderGrad/Fall2023/ASTR3510/Hot_Pixe_FInal' +'Flux_Time.png', dpi=300, bbox_inches='tight')

plt.show()

