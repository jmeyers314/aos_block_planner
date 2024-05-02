import re

import astropy.units as u
import galsim
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytz
from astroplan import Observer
from astropy.utils.masked import Masked
from astropy.coordinates import AltAz, Angle, SkyCoord, get_body
from astropy.table import QTable, vstack
from astropy.time import Time

RUBIN = Observer.at_site('cerro pachon')


def almanac(day, verbose=False):
    """Get sunset, sunrise info on day

    Paramters
    ---------
    day : str
        "YYYY-MM-DD" format day
    """
    noon_cp = Time(day)+15*u.h
    cptz = pytz.timezone("America/Santiago")
    pttz = pytz.timezone("US/Pacific")

    out = {
        'noon_cp': noon_cp,
        'sunset': {},
        'sunrise': {},
    }
    for alt in [0, -6, -12, -18]:
        out['sunset'][alt] = RUBIN.sun_set_time(noon_cp, horizon=alt*u.deg, which='next')
    for alt in [0, -6, -12, -18]:
        out['sunrise'][alt] = RUBIN.sun_rise_time(noon_cp, horizon=alt*u.deg, which='next')
    out['moonrise'] = RUBIN.moon_rise_time(noon_cp, horizon=0*u.deg, which='next')
    out['moonset'] = RUBIN.moon_set_time(noon_cp, horizon=0*u.deg, which='next')


    if verbose:
        print("   alt      America/Santiago                     UTC                      PT")
        print()
        print("sunset")
        for alt, time in out['sunset'].items():
            time_utc = time.to_datetime(timezone=pytz.utc)
            time_cp = time_utc.astimezone(cptz)
            time_pt = time_utc.astimezone(pttz)
            print(f"   {alt:3d}   {time_cp.strftime('%Y-%m-%d %H:%M:%S')}     {time_utc.strftime('%Y-%m-%d %H:%M:%S')}     {time_pt.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("sunrise")
        for alt, time in out['sunrise'].items():
            time_utc = time.to_datetime(timezone=pytz.utc)
            time_cp = time_utc.astimezone(cptz)
            time_pt = time_utc.astimezone(pttz)
            print(f"   {alt:3d}   {time_cp.strftime('%Y-%m-%d %H:%M:%S')}     {time_utc.strftime('%Y-%m-%d %H:%M:%S')}     {time_pt.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("moonrise")
        time_utc = out['moonrise'].to_datetime(timezone=pytz.utc)
        time_cp = time_utc.astimezone(cptz)
        time_pt = time_utc.astimezone(pttz)
        print(f"   {0:3d}   {time_cp.strftime('%Y-%m-%d %H:%M:%S')}     {time_utc.strftime('%Y-%m-%d %H:%M:%S')}     {time_pt.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("moonset")
        time_utc = out['moonset'].to_datetime(timezone=pytz.utc)
        time_cp = time_utc.astimezone(cptz)
        time_pt = time_utc.astimezone(pttz)
        print(f"   {0:3d}   {time_cp.strftime('%Y-%m-%d %H:%M:%S')}     {time_utc.strftime('%Y-%m-%d %H:%M:%S')}     {time_pt.strftime('%Y-%m-%d %H:%M:%S')}")
    return out


def position_angle(coord, obstime, location=RUBIN.location):
    """Compute rotator offset aligning camera +Y_CCS with ICRF south

    Parameters
    ----------
    coord : SkyCoord
        Boresight coordinate
    obstime : Time
        Time of observation

    Returns
    -------
    q : Angle
        Rotator angle.
    """
    try:
        coord.obstime=obstime
        coord.location=location
    except AttributeError:
        pass

    # Get point a little closer to zenith
    towards_zenith = SkyCoord(
        alt=coord.altaz.alt+10*u.arcsec,
        az=coord.altaz.az,
        frame=AltAz,
        obstime=obstime,
        location=location
    )

    # Note, for maximum precision, can't use coord.position_angle(towards_zenith)
    # since that computes the angle projected onto the celestial sphere, and we
    # want the angle projected onto AltAz.  AltAz isn't a simple spherical rotation,
    # there are stretches and skews too so milliarcsecond errors exist if computing
    # the angle on the ICRS sphere.  Instead, we use GalSim spherical trig and
    # compute on the AltAz sphere.

    # Get point a little closer to NCP
    towards_ncp = SkyCoord(
        ra=coord.icrs.ra,
        dec=coord.icrs.dec+10*u.arcmin,
        obstime=obstime,
        location=location
    )

    scalar = np.broadcast(
        coord.altaz.az.rad,
        towards_ncp.altaz.az.rad,
        towards_zenith.altaz.az.rad,
    ).ndim == 0

    qs = []
    for caz, calt, naz, nalt, zaz, zalt in zip(*np.broadcast_arrays(
        np.atleast_1d(coord.altaz.az.rad),
        np.atleast_1d(coord.altaz.alt.rad),
        np.atleast_1d(towards_ncp.altaz.az.rad),
        np.atleast_1d(towards_ncp.altaz.alt.rad),
        np.atleast_1d(towards_zenith.altaz.az.rad),
        np.atleast_1d(towards_zenith.altaz.alt.rad),
    )):
        cc0 = galsim.CelestialCoord(float(caz)*galsim.radians, float(calt)*galsim.radians)
        cc1 = galsim.CelestialCoord(float(naz)*galsim.radians, float(nalt)*galsim.radians)
        cc2 = galsim.CelestialCoord(float(zaz)*galsim.radians, float(zalt)*galsim.radians)
        qs.append(Angle(cc0.angleBetween(cc1, cc2).deg * u.deg).wrap_at(180*u.deg))
    if scalar:
        return qs[0]
    else:
        return u.Quantity(qs)


def rtp_to_rsp(rtp, coord, obstime):
    """Compute rotSkyPos from rotTelPos

    Parameters
    ----------
    rtp : Angle
        Rotator position in degrees
    coord : SkyCoord
        ICRF boresight in degrees
    obstime : Time
        time of observation

    Returns
    -------
    rsp : Angle
        rotSkyPos
    """
    q = position_angle(coord, obstime)
    return Angle(270*u.deg + rtp - q).wrap_at(180*u.deg)


def rsp_to_rtp(rsp, coord, obstime):
    """Compute rotSkyPos from rotTelPos

    Parameters
    ----------
    rsp : Angle
        Rotator position in degrees
    coord : SkyCoord
        ICRF boresight in degrees
    obstime : Time
        time of observation

    Returns
    -------
    rtp : Angle
        rotTelPos
    """
    q = position_angle(coord, obstime)
    return Angle(rsp + q - 270*u.deg).wrap_at(180*u.deg)


class Block:
    shift_dtype = np.dtype([('dz', '<f8'), ('dx', '<f8'), ('dy', '<f8')])
    rot_dtype = np.dtype([('rx', '<f8'), ('ry', '<f8')])
    block_schema = QTable(
        dtype=[
            ("block_name", "<U"),
            ("band", "<U"),
            ("exptime", "<f8"),
            ("elaptime", "<f8"),
            ("focusZ", "<f8"),
            ("m2_shift", shift_dtype),
            ("m2_rot", rot_dtype),
            ('cam_shift', shift_dtype),
            ("cam_rot", rot_dtype),
            ("m1m3_bend", "<f8", (20,)),
            ("m2_bend", "<f8", (20,))
        ],
        units={
            "exptime": u.s,
            "elaptime": u.s,
            "focusZ": u.mm,
            "m2_shift": u.micron,
            "m2_rot": u.arcsec,
            "cam_shift": u.micron,
            "cam_rot": u.arcsec,
            "m1m3_bend": u.micron,
            "m2_bend": u.micron
        }
    )

    def __init__(
        self,
        name: str,
        band: str = "i",
        exptime: u.Quantity = 30*u.s,
        elaptime: u.Quantity = 45*u.s,
        focusZ: u.Quantity = 0*u.mm,
        n: int = 1,
        m2_shift: np.ndarray = np.array((0, 0, 0), dtype=shift_dtype)*u.mm,
        m2_rot: np.ndarray = np.array((0, 0), dtype=rot_dtype)*u.arcsec,
        cam_shift: np.ndarray = np.array((0, 0, 0), dtype=shift_dtype)*u.mm,
        cam_rot: np.ndarray = np.array((0, 0), dtype=rot_dtype)*u.arcsec,
        m1m3_bend: np.ndarray = np.zeros(20)*u.mm,
        m2_bend: np.ndarray = np.zeros(20)*u.mm,
        **kwargs
    ):
        self.name = name,
        self.band = band,
        self.n = n
        self._table = self._initial_table()
        self._table["exptime"] = exptime
        self._table["elaptime"] = elaptime
        self._table["focusZ"] = focusZ
        self._table["m2_shift"] = m2_shift
        self._table["m2_rot"] = m2_rot
        self._table["cam_shift"] = cam_shift
        self._table["cam_rot"] = cam_rot
        self._table["m1m3_bend"] = m1m3_bend
        self._table["m2_bend"] = m2_bend

        for k, v in kwargs.items():
            # M1M3 matching
            if (match := re.match(r"^m1m3_b(\d+)$", k)):
                mode = int(match.group(1))
                if mode > 0 and mode < 21:
                    self._table["m1m3_bend"][:, mode-1] = v
                else:
                    raise ValueError(f"Invalid mode {mode}")
            # M2 matching
            elif (match := re.match(r"^m2_b(\d+)$", k)):
                mode = int(match.group(1))
                if mode > 0 and mode < 21:
                    self._table["m2_bend"][:, mode-1] = v
                else:
                    raise ValueError(f"Invalid mode {mode}")
            # M2 shift matching
            elif (match := re.match(r"^m2_shift_(dz|dx|dy)$", k)):
                mode = match.group(1)
                if mode in ["dz", "dx", "dy"]:
                    self._table["m2_shift"][mode] = v
                else:
                    raise ValueError(f"Invalid mode {mode}")
            # M2 rot matching
            elif (match := re.match(r"^m2_rot_(rx|ry)$", k)):
                mode = match.group(1)
                if mode in ["rx", "ry"]:
                    self._table["m2_rot"][mode] = v
                else:
                    raise ValueError(f"Invalid mode {mode}")
            # Camera shift matching
            elif (match := re.match(r"^cam_shift_(dz|dx|dy)$", k)):
                mode = match.group(1)
                if mode in ["dz", "dx", "dy"]:
                    self._table["cam_shift"][mode] = v
                else:
                    raise ValueError(f"Invalid mode {mode}")
            # Camera rot matching
            elif (match := re.match(r"^cam_rot_(rx|ry)$", k)):
                mode = match.group(1)
                if mode in ["rx", "ry"]:
                    self._table["cam_rot"][mode] = v
                else:
                    raise ValueError(f"Invalid mode {mode}")
            else:
                # Anything else, just append to the table
                self._table[k] = v

    def duration(self):
        return np.sum(self._table["elaptime"])

    def _initial_table(self):
        table = QTable(self.block_schema)
        for idx in range(self.n):
            table.add_row({
                "block_name": self.name,
                "band": self.band,
                "exptime": 0.0*u.s,
                "elaptime": 0.0*u.s,
                "focusZ": 0*u.mm,
                "m2_shift": np.array((0, 0, 0), dtype=self.shift_dtype)*u.mm,
                "m2_rot": np.array((0, 0), dtype=self.rot_dtype)*u.arcsec,
                "cam_shift": np.array((0, 0, 0), dtype=self.shift_dtype)*u.mm,
                "cam_rot": np.array((0, 0), dtype=self.rot_dtype)*u.arcsec,
                "m1m3_bend": np.zeros(20)*u.mm,
                "m2_bend": np.zeros(20)*u.mm
            })
        return table


def schedule_blocks(blocks, start_time):
    schedule = QTable(Block.block_schema)
    for block in blocks.values():
        schedule = vstack([schedule, block._table])
    n = len(schedule)

    for name in ['ra', 'dec', 'rsp', 'rtp', 'alt', 'az']:
        if name not in schedule.colnames:
            schedule[name] = Masked(np.full(n, np.nan)*u.deg, mask=True)

    # First fill in obstime
    schedule['obstime'] = start_time + np.cumsum(schedule['elaptime'])
    schedule['obstime'].format='iso'

    # Where ra/dec are missing, compute from alt/az
    w = schedule['ra'].mask & ~schedule['alt'].mask
    if np.any(w):
        coord = SkyCoord(
            schedule['az'][w],
            schedule['alt'][w],
            frame=AltAz(
                obstime=schedule['obstime'][w],
                location=RUBIN.location
            )
        )
        schedule['ra'][w] = coord.icrs.ra
        schedule['dec'][w] = coord.icrs.dec

    # Where rsp is missing, compute from rtp
    w = schedule['rsp'].mask & ~schedule['rtp'].mask
    if np.any(w):
        coord = SkyCoord(schedule['ra'][w], schedule['dec'][w])
        schedule['rsp'][w] = rtp_to_rsp(
            schedule['rtp'][w],
            coord=SkyCoord(
                schedule['ra'][w],
                schedule['dec'][w],
                obstime=schedule['obstime'][w],
                location=RUBIN.location
            ),
            obstime=schedule['obstime'][w]
        )

    # Where ra/dec/rsp are still missing, assume it's the same as previous (i.e., tracking)
    prev_ra = None
    prev_dec = None
    prev_rsp = None
    for row in schedule:
        if not row['ra'].mask:
            prev_ra = row['ra']
            prev_dec = row['dec']
        else:
            row['ra'] = prev_ra
            row['dec'] = prev_dec
        if not row['rsp'].mask:
            prev_rsp = row['rsp']
        else:
            row['rsp'] = prev_rsp

    # We should now be able to go back and fill in missing alt/az/rtp
    w = schedule['alt'].mask & ~schedule['ra'].mask
    if np.any(w):
        coord = SkyCoord(
            schedule['ra'][w],
            schedule['dec'][w],
            obstime=schedule['obstime'][w],
            location=RUBIN.location
        )
        schedule['alt'][w] = coord.altaz.alt
        schedule['az'][w] = coord.altaz.az

    w = schedule['rtp'].mask & ~schedule['rsp'].mask
    if np.any(w):
        coord = SkyCoord(schedule['ra'][w], schedule['dec'][w])
        schedule['rtp'][w] = rsp_to_rtp(
            schedule['rsp'][w],
            coord=SkyCoord(
                schedule['ra'][w],
                schedule['dec'][w],
                obstime=schedule['obstime'][w],
                location=RUBIN.location
            ),
            obstime=schedule['obstime'][w]
        )

    return schedule


class PeekSky:
    def __init__(self):
        self.gaia = QTable.read("density.ecsv")
        n_points = len(self.gaia)
        self.n_side = int(np.sqrt(n_points / 12))
        self.indices = np.arange(hp.nside2npix(self.n_side))
        self.gaia_ra, self.gaia_dec = hp.pix2ang(self.n_side, self.indices, lonlat=True)

    @staticmethod
    def azalt_to_postel(az, alt):
        r = 90-alt
        y = np.cos(np.deg2rad(az))*r
        x = -np.sin(np.deg2rad(az))*r
        return x, y

    def peek_sky(self, time, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='k')
            ax.set_facecolor('k')
        else:
            fig = ax.figure

        frame = AltAz(obstime=time, location=RUBIN.location)

        # Draw altaz grid
        az = np.linspace(0, 2*np.pi, 100)
        for zen in np.arange(0, 90, 15):
            y, x = zen*np.cos(az), -zen*np.sin(az)
            ax.plot(x, y, c='g', lw=0.2)
        zen = np.linspace(0, 90)
        for az in np.arange(0, 2*np.pi, np.pi/4):
            y, x = zen*np.cos(az), -zen*np.sin(az)
            ax.plot(x, y, c='g', lw=(1.0 if az == 0 else 0.2))
        ra = np.linspace(0, 2*np.pi, 100)
        for dec in np.arange(-90, 90, 15):
            c = SkyCoord(ra*u.rad, dec*u.deg)
            c = c.transform_to(frame)
            zen = 90-c.altaz.alt.deg
            zen[zen>90] = np.nan
            y, x = zen*np.cos(c.altaz.az), -zen*np.sin(c.altaz.az)
            ax.plot(x, y, c='c', lw=0.2)
        dec = np.linspace(-90, 90, 100)
        for ra in np.arange(0, 2*np.pi, np.pi/4):
            c = SkyCoord(ra*u.rad, dec*u.deg)
            c = c.transform_to(frame)
            zen = 90-c.altaz.alt.deg
            zen[zen>90] = np.nan
            y, x = zen*np.cos(c.altaz.az), -zen*np.sin(c.altaz.az)
            ax.plot(x, y, c='c', lw=(1.0 if ra == 0.0 else 0.2))

        # Draw sun
        sunCoords = get_body('sun', time)
        sunCoords.location = RUBIN.location
        if sunCoords.altaz.alt.value > 0:
            ax.text(
                *self.azalt_to_postel(
                    sunCoords.altaz.az.deg,
                    sunCoords.altaz.alt.deg
                ),
                "☉",
                c='y',
                fontsize=25, weight='bold'
            )

        # Draw moon
        moonCoords = get_body('moon', time)
        moonCoords.location = RUBIN.location

        if moonCoords.altaz.alt.value > 0:
            ax.text(
                *self.azalt_to_postel(
                    moonCoords.altaz.az.deg,
                    moonCoords.altaz.alt.deg
                ),
                "☽︎",
                c='c',
                fontsize=25, weight='bold'
            )

        # Draw Jupiter and Venus
        jupiterCoords = get_body('jupiter', time)
        jupiterCoords.location = RUBIN.location
        if jupiterCoords.altaz.alt.value > 0:
            ax.text(
                *self.azalt_to_postel(
                    jupiterCoords.altaz.az.deg,
                    jupiterCoords.altaz.alt.deg
                ),
                "♃",
                c='r',
                fontsize=15, weight='bold'
            )

        venusCoords = get_body('venus', time)
        venusCoords.location = RUBIN.location
        if venusCoords.altaz.alt.value > 0:
            ax.text(
                *self.azalt_to_postel(
                    venusCoords.altaz.az.deg,
                    venusCoords.altaz.alt.deg
                ),
                "♀",
                c='g',
                fontsize=15, weight='bold'
            )

        # Cardinal directions
        ax.text(*self.azalt_to_postel(0, -5), "N", c='r', fontsize=15, verticalalignment='center', horizontalalignment='center')
        ax.text(*self.azalt_to_postel(90, -5), "E", c='r', fontsize=15, verticalalignment='center', horizontalalignment='center')
        ax.text(*self.azalt_to_postel(180, -5), "S", c='r', fontsize=15, verticalalignment='center', horizontalalignment='center')
        ax.text(*self.azalt_to_postel(270, -5), "W", c='r', fontsize=15, verticalalignment='center', horizontalalignment='center')
        ax.text(*self.azalt_to_postel(45, -7), "45", c='g', fontsize=10, verticalalignment='center', horizontalalignment='center')
        ax.text(*self.azalt_to_postel(135, -7), "135", c='g', fontsize=10, verticalalignment='center', horizontalalignment='center')
        ax.text(*self.azalt_to_postel(225, -7), "225", c='g', fontsize=10, verticalalignment='center', horizontalalignment='center')
        ax.text(*self.azalt_to_postel(315, -7), "315", c='g', fontsize=10, verticalalignment='center', horizontalalignment='center')

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        xs = np.linspace(-100, 100, 700)
        xs, ys = np.meshgrid(xs, xs)
        alt = np.deg2rad(90 - np.hypot(xs, ys))
        w = alt >= 0
        az = np.arctan2(-xs, ys)
        coords = SkyCoord(az*u.rad, alt*u.rad, frame=frame)
        ra = coords.icrs.ra.deg
        dec = coords.icrs.dec.deg
        idxs = hp.ang2pix(self.n_side, ra, dec, lonlat=True)

        density = self.gaia[idxs]['density']
        density[~w] = np.nan

        ax.imshow(density, extent=[-100, 100, -100, 100], origin='lower')

        def xystr(x, y):
            alt = 90 - np.hypot(x,y)
            az = np.rad2deg(np.arctan2(-x, y))%360
            c = SkyCoord(az*u.deg, alt*u.deg, frame=frame)
            α = c.icrs.ra.deg
            δ = c.icrs.dec.deg
            return f"{alt=:.1f} {az=:.1f} {α=:.1f} {δ=:.1f}"

        def hover(event):
            if event.inaxes == ax:
                ax.format_coord = xystr

        fig.canvas.mpl_connect('motion_notify_event', hover)
        fig.tight_layout()

        return fig, ax
