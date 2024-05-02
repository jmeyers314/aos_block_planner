import healpy as hp
import numpy as np
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import contextlib
import logging
from pathlib import Path
from astropy.table import Table


@contextlib.contextmanager
def suppress_logging(name=None, level=logging.CRITICAL):
    logger = logging.getLogger(name)
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)

n_points = 42000
n_side = int(np.sqrt(n_points / 12))
indices = np.arange(hp.nside2npix(n_side))
ra, dec = hp.pix2ang(n_side, indices, lonlat=True)
print(n_side, len(ra))

# r = None
# maxdist = []
# nstar = []
fn = Path("density_checkpoint.ecsv")
if fn.exists():
    table = Table.read(fn)
    maxdist = list(table['maxdist'])
    nstar = list(table['nstar'])
else:
    table = Table()
    maxdist = []
    nstar = []
Gaia.ROW_LIMIT=100

for ra_, dec_ in zip(tqdm(ra[len(table):]), dec[len(table):]):
    coord = SkyCoord(ra_*u.deg, dec_*u.deg)
    with suppress_logging("astroquery"):
        job = Gaia.cone_search_async(coord, radius=0.1*u.deg)
        r = job.get_results()
    maxdist.append(np.max(r['dist']))
    nstar.append(len(r))

    density = np.array(nstar)/(np.pi*np.array(maxdist)**2)

    table = Table()
    table['ra'] = ra[:len(density)]*u.deg
    table['dec'] = dec[:len(density)]*u.deg
    table['nstar'] = nstar
    table['maxdist'] = maxdist*u.deg
    table['density'] = density
    table.write("density_checkpoint.ecsv", overwrite=True)

density = np.array(nstar)/(np.pi*np.array(maxdist)**2)
table = Table()
table['ra'] = (ra*u.rad).to(u.deg)
table['dec'] = (dec*u.rad).to(u.deg)
table['density'] = density
table.write("density.ecsv", overwrite=True)
