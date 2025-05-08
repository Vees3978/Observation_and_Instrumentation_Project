import astropy.units as u
import matplotlib.pyplot as plt 
import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats 
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from glob import glob
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats

class flare_finder:

    GJ182_COORDS_J2000 = ['04h59m34.8342878424s +01d47m00.669818328s']
    REF_STAR_0_COORDS_J2000 = ['04h59m30.9509965632	+01d45m24.447571200s'] #TYC 85-1241-1

    APOLLO_PATH = 'data/11_29_23/Apollo_r'
    ARTEMIS_PATH = 'data/11_29_23/Artemis_g'
    LETO_PATH = 'data/11_29_23/Leto_i'

    FWHM_ESTIMATE = 8.0

    def __init__(self):
        self.load_files()
        self.find_initial_target_coords()
        self.update_apetures()
        self.run_photometry()
        self.draw_figure()

    def load_files(self):
        apollo_files = glob(self.APOLLO_PATH + '/data/*.fits')
        artemis_files = glob(self.ARTEMIS_PATH + '/data/*.fits')
        leto_files = glob(self.LETO_PATH + '/data/*.fits')

        apollo_dark_files = glob(self.APOLLO_PATH + '/noise/*-Dark-*.fit')
        artemis_dark_files = glob(self.ARTEMIS_PATH + '/noise/*-Dark-*.fit')
        leto_dark_files = glob(self.LETO_PATH + '/noise/*-Dark-*.fit')
        
        apollo_bias_files = glob(self.APOLLO_PATH + '/noise/*-Bias-*.fit')
        artemis_bias_files = glob(self.ARTEMIS_PATH + '/noise/*-Bias-*.fit')
        leto_bias_files = glob(self.LETO_PATH + '/noise/*-Bias-*.fit')

        self.apollo_data = self.append_fits_data(apollo_files)
        print("Loaded " + str(len(self.apollo_data)) + " data files from Apollo")
        self.artemis_data = self.append_fits_data(artemis_files)
        print("Loaded " + str(len(self.artemis_data)) + " data files from Artemis")
        self.leto_data = self.append_fits_data(leto_files)
        print("Loaded " + str(len(self.leto_data)) + " data files from Leto")
        
        self.apollo_darks = self.append_fits_data(apollo_dark_files)
        print("Loaded " + str(len(self.apollo_darks)) + " dark files from Apollo")
        self.artemis_darks = self.append_fits_data(artemis_dark_files)
        print("Loaded " + str(len(self.artemis_darks)) + " dark files from Artemis")
        self.leto_darks = self.append_fits_data(leto_dark_files)
        print("Loaded " + str(len(self.leto_darks)) + " dark files from Leto")
        
        self.apollo_biases = self.append_fits_data(apollo_bias_files)
        print("Loaded " + str(len(self.apollo_biases)) + " bias files from Apollo")
        self.artemis_biases = self.append_fits_data(artemis_bias_files)
        print("Loaded " + str(len(self.artemis_biases)) + " bias files from Artemis")
        self.leto_biases = self.append_fits_data(leto_bias_files)
        print("Loaded " + str(len(self.leto_biases)) + " bias files from Leto")
    
    def find_initial_target_coords(self):
        self.apollo_target_coords = self.get_pixel_coords(self.APOLLO_PATH, self.GJ182_COORDS_J2000)
        print("Pixel coords of GJ 182 from Apollo-r data is", self.apollo_target_coords)
        self.artemis_target_coords = self.get_pixel_coords(self.ARTEMIS_PATH, self.GJ182_COORDS_J2000)
        print("Pixel coords of GJ 182 from Artemis-g data is", self.artemis_target_coords)
        self.leto_target_coords = [354, 526] # TODO fix: self.get_pixel_coords(self.LETO_PATH, self.GJ182_COORDS_J2000) #WCS fail on ir data????
        print("Pixel coords of GJ 182 from Leto-i data is", self.leto_target_coords)

        self.apollo_ref_0_coords = self.get_pixel_coords(self.APOLLO_PATH, self.REF_STAR_0_COORDS_J2000)
        print("Pixel coords of REF STAR 0 from Apollo-r data is", self.apollo_ref_0_coords)
        self.artemis_ref_0_coords = self.get_pixel_coords(self.ARTEMIS_PATH, self.REF_STAR_0_COORDS_J2000)
        print("Pixel coords of REF STAR 0 from Artemis-g data is", self.artemis_ref_0_coords)
        self.leto_ref_0_coords = [450, 675] # TODO FIX
        print("Pixel coords of REF STAR 0 from Leto-i data is", self.leto_ref_0_coords)

    def update_apetures(self, index=0):
        apollo_stars = self.find_stars(self.apollo_data[index])
        artemis_stars = self.find_stars(self.artemis_data[index])
        leto_stars = self.find_stars(self.leto_data[index])

        self.apollo_target_coords = self.find_closest_coords(apollo_stars, self.apollo_target_coords)
        self.artemis_target_coords = self.find_closest_coords(artemis_stars, self.artemis_target_coords)
        self.leto_target_coords = self.find_closest_coords(leto_stars, self.leto_target_coords)

        self.apollo_ref_0_coords = self.find_closest_coords(apollo_stars, self.apollo_ref_0_coords)
        self.artemis_ref_0_coords = self.find_closest_coords(artemis_stars, self.artemis_ref_0_coords)
        self.leto_ref_0_coords = self.find_closest_coords(leto_stars, self.leto_ref_0_coords)

        self.apollo_star_apts, self.apollo_sky_apts = self.create_apetures(self.apollo_target_coords, self.apollo_ref_0_coords)
        print("Created " + str(len(self.apollo_star_apts)) + " star apetrues and " + str(len(self.apollo_sky_apts)) + " sky apetures for Apollo-r")
        self.artemis_star_apts, self.artemis_sky_apts = self.create_apetures(self.artemis_target_coords, self.artemis_ref_0_coords)
        print("Created " + str(len(self.artemis_star_apts)) + " star apetrues and " + str(len(self.artemis_sky_apts)) + " sky apetures for Artemis-g")
        self.leto_star_apts, self.leto_sky_apts = self.create_apetures(self.leto_target_coords, self.leto_ref_0_coords)
        print("Created " + str(len(self.leto_star_apts)) + " star apetrues and " + str(len(self.leto_sky_apts)) + " sky apetures for Leto-i")

    def run_photometry(self):
        print("Doing photometry on Apollo data...")
        self.apollo_target_star_fluxes, self.apollo_target_star_uncertainties, self.apollo_target_sky_fluxes = self.compute_light_curves(self.apollo_data, self.apollo_star_apts, self.apollo_sky_apts)
        print("Doing photometry on Artemis data...")
        self.artemis_target_star_fluxes, self.artemis_target_star_uncertainties, self.artemis_target_sky_fluxes = self.compute_light_curves(self.artemis_data, self.artemis_star_apts, self.artemis_sky_apts)
        print("Doing photometry on Leto data...")
        self.leto_target_star_fluxes, self.leto_target_star_uncertainties, self.leto_target_sky_fluxes = self.compute_light_curves(self.leto_data, self.leto_star_apts, self.leto_sky_apts)

    def draw_figure(self):
        plt.style.use('dark_background')

        self.fig, self.axs = plt.subplots(3, 3)

        self.fig.tight_layout(pad=1.0)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(14)

        self.axs[0, 0].set_title('Apollo r\' frame')
        self.axs[1, 0].set_title('Artemis g\' frame')
        self.axs[2, 0].set_title('Leto i\' frame')

        self.axs[0, 1].set_title('Apollo r\' Relative Flux vs Exposure Count')
        self.axs[1, 1].set_title('Artemis g\' Relative Flux vs Exposure Count')
        self.axs[2, 1].set_title('Leto i\' Relative Flux vs Exposure Count')

        self.axs[0, 2].set_title('Apollo r\' SNR vs Exposure Count')
        self.axs[1, 2].set_title('Artemis g\' SNR vs Exposure Count')
        self.axs[2, 2].set_title('Leto i\' SNR vs Exposure Count')
        
        self.plot_frame(self.axs[0, 0], self.apollo_data)
        self.plot_frame(self.axs[1, 0], self.artemis_data)
        self.plot_frame(self.axs[2, 0], self.leto_data)
  
        self.plot_apetures(self.axs[0, 0], self.apollo_star_apts, self.apollo_sky_apts)
        self.plot_apetures(self.axs[1, 0], self.artemis_star_apts, self.artemis_sky_apts)
        self.plot_apetures(self.axs[2, 0], self.leto_star_apts, self.leto_sky_apts)

        self.plot_relative_fluxes(self.axs[0, 1], self.apollo_target_star_fluxes)
        self.plot_relative_fluxes(self.axs[1, 1], self.artemis_target_star_fluxes)
        self.plot_relative_fluxes(self.axs[2, 1], self.leto_target_star_fluxes)

        plt.show()

    def append_fits_data(self, glob_list):
        fits_list = []
        for f in glob_list:
            fits_list.append(fits.open(f)[0].data)
        return fits_list
    
    def get_pixel_coords(self, telescope, stellar_coords):
        sky_coord = SkyCoord(stellar_coords, unit=(u.hourangle, u.deg), obstime="J2000")
        wcs_header = fits.open(telescope + '/wcs.fits')[0].header
        w = WCS(wcs_header)
        x, y = w.world_to_pixel(sky_coord)
        return [x[0], y[0]]

    def find_stars(self, data):
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        #print((mean, median, std))  
        daofind = DAOStarFinder(fwhm=self.FWHM_ESTIMATE, threshold=5.*std) # Stars were closer to 8px fwhm (breaks if 3px)
        sources = daofind(data - median)  
        for col in sources.colnames:  
            if col not in ('id', 'npix'):
                sources[col].info.format = '%.2f'
        #sources.pprint(max_width=200)  
        #print("Found", len(sources), "stars.")

        return sources
    
    def find_closest_coords(self, stars, target):
        min_dis = 100000
        next_target = target

        for star in stars:
            x = star['xcentroid']
            y = star['ycentroid']

            dis = np.sqrt((x - target[0])**2 + (y - target[1])**2)

            if dis < min_dis:
                min_dis = dis
                next_target = [x, y]

        return next_target

    def create_apetures(self, target, ref):
        star_apertures = []
        sky_apertures = []

        star_apertures.append(CircularAperture(target, r=20)) # Target star is index 0
        sky_apertures.append(CircularAnnulus(target, r_in=25, r_out=45))

        star_apertures.append(CircularAperture(ref, r=20))
        sky_apertures.append(CircularAnnulus(ref, r_in=25, r_out=45))

        return star_apertures, sky_apertures

    def compute_light_curves(self, data, star_apertures, sky_apertures):
        fluxes = np.array(data)

        all_star_fluxes = []
        all_star_uncertainties = []
        all_sky_fluxes = []

        for i in range(len(star_apertures)):
            star_fluxes, star_uncertainties, sky_fluxes = self.__compute_light_curve(fluxes, star_apertures[i], sky_apertures[i])
            all_star_fluxes.append(star_fluxes)
            all_star_uncertainties.append(star_uncertainties)
            all_sky_fluxes.append(sky_fluxes)

        return all_star_fluxes, all_star_uncertainties, all_sky_fluxes

    def __compute_light_curve(self, fluxes, star_aperture, sky_aperture):
        nexposures, nrows, ncols = fluxes.shape

        # create empty arrays to store results 
        star_fluxes = np.zeros(nexposures)
        star_uncertainties = np.zeros(nexposures)
        sky_fluxes = np.zeros(nexposures)

        recompute_freq = 50
        recompute_count = 0

        # loop through exposures
        for i in range(nexposures):
            # if recompute_count > recompute_freq:
            #     self.update_apetures(i)
            #     recompute_count = 0

            # recompute_count += 1
            
            # extract the images for one particular exposure
            f_image = fluxes[i, :, :]
            #u_image = self.uncertainties[i, :, :] # pretty certin this was not supposed to be flux (as it was originally)
            
            # do photometry on an image using an aperture
            aperture_flux, aperture_uncertainty = star_aperture.do_photometry(f_image)
            # calculate statistics on the pixels in the sky aperture
            sky_stats = ApertureStats(f_image, sky_aperture)

            # calculate an estimate of the sky per pixel
            sky_per_pixel = sky_stats.median

            # calculate how much sky flux must be in our star aperture
            sky_flux = star_aperture.area*sky_per_pixel
            sky_fluxes[i] = sky_flux
            
            # subtract the sky from the aperture flux 
            star_fluxes[i] = aperture_flux[0] - sky_flux
            #star_uncertainties[i] = aperture_uncertainty[0]

        return star_fluxes, star_uncertainties, sky_fluxes

    def plot_frame(self, axis, data_list):
        display_index = len(data_list) - 1
        axis.imshow(data_list[display_index], vmin=np.percentile(data_list[display_index], 1), vmax=np.percentile(data_list[display_index], 99))

    def plot_apetures(self, axis, star_apertures, sky_apertures):
        for i, star_aperture in enumerate(star_apertures):
            if i == 0:
                star_aperture.plot(color='red', ax=axis)
            else:
                star_aperture.plot(color='white', ax=axis)
        
        for i, sky_aperture in enumerate(sky_apertures):
            if i == 0:
                sky_aperture.plot(color='red', linestyle='--', ax=axis)
            else:
                sky_aperture.plot(color='white', linestyle='--', ax=axis)

    def plot_relative_fluxes(self, axis, star_fluxes):
        relative_flux_target = star_fluxes[0]/np.median(star_fluxes[0])
        relative_flux_ref_avg = np.zeros(len(star_fluxes[0]))

        for i, flux in enumerate(star_fluxes):
            if i > 0:
                relative_flux_ref_avg += flux/np.median(flux)

        relative_flux_ref_avg /= len(star_fluxes) - 1
        
        axis.plot(relative_flux_target - relative_flux_ref_avg)

ff = flare_finder()