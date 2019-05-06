import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

table = Table.read('bright_fgk-result.fits')
table['coords'] = SkyCoord(ra=table['ra'], dec=table['dec'])

sep_limit = 2. * u.pc
bp_limit = 0.5
rp_limit = 0.5

pairs = Table(names=('source_id1', 'source_id2', 'ra1', 'ra2', 'dec1', 'dec2',
                     'plx1', 'plx_error1', 'plx2', 'plx_error2', 
                     'pmra1', 'pmra_error1', 'pmra2', 'pmra_error2', 
                     'pmdec1', 'pmdec_error1', 'pmdec2', 'pmdec_error2',
                     'bp1', 'bp2', 'rp1', 'rp2', 'g1', 'g2', 'mg1', 'mg2',
                     'delta_distance', 'delta_distance_error', 'delta_pm', 'delta_pm_error',
                     'angsep'),
             dtype=('i8', 'i8', 'f8', 'f8', 'f8', 'f8',
                    'f8', 'f8', 'f8', 'f8', 
                    'f8', 'f8', 'f8', 'f8', 
                    'f8', 'f8', 'f8', 'f8', 
                    'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                    'f8', 'f8', 'f8', 'f8', 'f8'))
                    
def row_exists(pairs, s1, s2):
    mask1 = (pairs['source_id1'] == s1) & (pairs['source_id2'] == s2)
    mask2 = (pairs['source_id1'] == s2) & (pairs['source_id2'] == s1)
    if np.sum(mask1) > 0 or np.sum(mask2) > 0:
        return True
    else:
        return False
        
for star in tqdm(table):
    distance = (star['parallax'] * u.mas).to(u.pc, equivalencies=u.parallax()) # distance to star1
    nearby_stars = table[star['coords'].separation(table['coords']) <= sep_limit / distance * u.rad] 
    nearby_stars = nearby_stars[nearby_stars['source_id'] != star['source_id']] # eliminate self-match
    if len(nearby_stars) == 0: 
        #print('no nearby stars found!') 
        continue
    similar_mags = (np.abs(star['phot_bp_mean_mag'] - nearby_stars['phot_bp_mean_mag']) <= bp_limit) \
                    & (np.abs(star['phot_rp_mean_mag'] - nearby_stars['phot_rp_mean_mag']) <= rp_limit)
    nearby_stars = nearby_stars[similar_mags] # apply magnitude cuts
    if len(nearby_stars) == 0: 
        #print('all stars failed magnitude cut')
        continue
    d2 = nearby_stars['parallax'].to(u.pc, equivalencies=u.parallax()) # distances to star2s
    delta_d = np.abs(distance - d2) # distance difference
    delta_d_err = np.sqrt(distance**2/star['parallax_over_error']**2 + d2**2/nearby_stars['parallax_over_error']**2)
    similar_plx = delta_d <= 3. * delta_d_err + 2.*sep_limit
    nearby_stars = nearby_stars[similar_plx] # apply parallax cut
    if len(nearby_stars) == 0: 
        #print('all stars failed parallax cut') 
        continue
    delta_pm = np.sqrt((star['pmra'] - nearby_stars['pmra'])**2 + (star['pmdec'] - nearby_stars['pmdec'])**2)
    delta_pm_err = np.sqrt((star['pmra_error']**2 + nearby_stars['pmra_error']**2)*(star['pmra'] - nearby_stars['pmra'])**2
                           + (star['pmdec_error']**2 + nearby_stars['pmdec_error']**2)*(star['pmdec'] - nearby_stars['pmdec'])**2) \
                    * 1./delta_pm
    separation = star['coords'].separation(nearby_stars['coords']).to('arcsec')
    delta_pm_orbit = 0.44 * star['parallax']**1.5 / np.sqrt(separation/u.arcsec)  * u.mas / u.yr
    similar_pm = delta_pm <= 3. * delta_pm_err + delta_pm_orbit
    nearby_stars = nearby_stars[similar_pm] # apply proper motion cut
    if len(nearby_stars) == 0: 
        #print('all stars failed proper motion cut') 
        continue
    for star2 in nearby_stars:
        if not row_exists(pairs, star['source_id'], star2['source_id']):
            d2 = (star2['parallax'] * u.mas).to(u.pc, equivalencies=u.parallax())
            delta_d = np.abs(distance - d2) # distance difference
            delta_d_err = np.sqrt(distance**2/star['parallax_over_error']**2 + d2**2/star2['parallax_over_error']**2)
            delta_pm = np.sqrt((star['pmra'] - star2['pmra'])**2 + (star['pmdec'] - star2['pmdec'])**2)
            delta_pm_err = np.sqrt((star['pmra_error']**2 + star2['pmra_error']**2)*(star['pmra'] - star2['pmra'])**2
                           + (star['pmdec_error']**2 + star2['pmdec_error']**2)*(star['pmdec'] - star2['pmdec'])**2) \
                    * 1./delta_pm 
            angsep = star['coords'].separation(star2['coords']).to('arcsec')
            pairs.add_row((star['source_id'], star2['source_id'], star['ra'], star2['ra'], star['dec'], star2['dec'],
                        star['parallax'], star['parallax_error'], star2['parallax'], star2['parallax_error'],
                        star['pmra'], star['pmra_error'], star2['pmra'], star2['pmra_error'],
                        star['pmdec'], star['pmdec_error'], star2['pmdec'], star2['pmdec_error'],
                        star['phot_bp_mean_mag'], star2['phot_bp_mean_mag'], star['phot_rp_mean_mag'], star2['phot_rp_mean_mag'],
                        star['phot_g_mean_mag'], star2['phot_g_mean_mag'], star['mg'], star2['mg'],
                        delta_d, delta_d_err, delta_pm, delta_pm_err, angsep))

pairs.write('pairs.fits')
print("{0} pairs found.".format(len(pairs)))