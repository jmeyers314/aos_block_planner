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


# RUBIN = Observer.at_site('cerro pachon')
# Use OpSim values
RUBIN = Observer(
    longitude=-70.7494*u.deg, latitude=-30.2444*u.deg,
    elevation=2650.0*u.m, name="LSST",
    timezone="Chile/Continental",
    pressure=750.0*u.mBa,
    temperature=11.5*u.deg_C,
    relative_humidity=0.4
)


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
    for el in [0, -6, -12, -18]:
        out['sunset'][el] = RUBIN.sun_set_time(
            noon_cp,
            horizon=el*u.deg,
            which='next'
        )
    for el in [-18, -12, -6, 0]:
        out['sunrise'][el] = RUBIN.sun_rise_time(
            noon_cp,
            horizon=el*u.deg,
            which='next'
        )
    out['moonrise'] = RUBIN.moon_rise_time(
        out['sunrise'][0],
        horizon=0*u.deg,
        which='previous'
    )
    out['moonset'] = RUBIN.moon_set_time(
        out['moonrise'],
        horizon=0*u.deg,
        which='next'
    )
    out['prev_moonset'] = RUBIN.moon_set_time(
        out['moonrise'],
        horizon=0*u.deg,
        which='previous'
    )
    out['prev_moonrise'] = RUBIN.moon_rise_time(
        out['prev_moonset'],
        horizon=0*u.deg,
        which='previous'
    )
    out['next_moonrise'] = RUBIN.moon_rise_time(
        out['moonset'],
        horizon=0*u.deg,
        which='next'
    )
    out['next_moonset'] = RUBIN.moon_set_time(
        out['next_moonrise'],
        horizon=0*u.deg,
        which='next'
    )

    if verbose:
        print("    el      America/Santiago                     UTC                      PT")
        print()
        print("sunset")
        for el, time in out['sunset'].items():
            time_utc = time.to_datetime(timezone=pytz.utc)
            time_cp = time_utc.astimezone(cptz)
            time_pt = time_utc.astimezone(pttz)
            print(f"   {el:3d}   {time_cp.strftime('%Y-%m-%d %H:%M:%S')}     {time_utc.strftime('%Y-%m-%d %H:%M:%S')}     {time_pt.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("sunrise")
        for el, time in out['sunrise'].items():
            time_utc = time.to_datetime(timezone=pytz.utc)
            time_cp = time_utc.astimezone(cptz)
            time_pt = time_utc.astimezone(pttz)
            print(f"   {el:3d}   {time_cp.strftime('%Y-%m-%d %H:%M:%S')}     {time_utc.strftime('%Y-%m-%d %H:%M:%S')}     {time_pt.strftime('%Y-%m-%d %H:%M:%S')}")
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

        print()
        print(f"Total dark time: {(out['sunrise'][-12]-out['sunset'][-12]).to(u.h):.2f} hours")
    return out


def pseudo_parallactic_angle(coord, obstime, location=RUBIN.location):
    """Compute the pseudo parallactic angle.

    The (traditional) parallactic angle is the angle zenith - coord - NCP
    where NCP is the true-of-date north celestial pole.  This function instead
    computes zenith - coord - NCP_ICRF where NCP_ICRF is the north celestial
    pole in the International Celestial Reference Frame.

    Parameters
    ----------
    coord : SkyCoord
        Boresight coordinate.  Any adjustments due to pressure, temperature,
        releative humidity, or wavelength should be applied to the coord
        object.
    obstime : Time
        Time of observation
    location : EarthLocation
        Location of telescope

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
    assert np.all(coord.obstime == obstime)
    assert np.all(coord.location == location)

    # Get point a little closer to zenith
    small_angle = 10*u.arcsec
    zalt = coord.altaz.alt + small_angle
    zaz = coord.altaz.az
    zdist = 90*u.deg - coord.altaz.alt
    if np.any(w := zdist < small_angle):
        # Overshoot zenith by small_angle
        zalt[w] = small_angle - zdist[w]
        zaz[w] = coord.altaz.az[w] + np.pi*u.rad
    towards_zenith = SkyCoord(
        alt=zalt,
        az=zaz,
        frame=AltAz,
        obstime=obstime,
        location=location,
        pressure=coord.pressure,
        temperature=coord.temperature,
        relative_humidity=coord.relative_humidity,
        obswl=coord.obswl
    )

    # Get point a little closer to NCP
    ndec = coord.icrs.dec + small_angle
    nra = coord.icrs.ra
    ndist = 90*u.deg - coord.icrs.dec
    if np.any(w:= ndist < small_angle):
        # Overshoot NCP by small_angle
        ndec[w] = small_angle - ndist[w]
        nra[w] = coord.icrs.ra[w] + np.pi*u.rad
    towards_ncp = SkyCoord(
        ra=nra,
        dec=ndec,
        obstime=obstime,
        location=location,
        pressure=coord.pressure,
        temperature=coord.temperature,
        relative_humidity=coord.relative_humidity,
        obswl=coord.obswl
    )

    return (
        coord.position_angle(towards_zenith)
        - coord.position_angle(towards_ncp)
    )


def rtp_to_rsp(rtp, coord, obstime, location=RUBIN.location):
    """Compute rotSkyPos from rotTelPos

    Parameters
    ----------
    rtp : Angle
        Rotator position
    coord : SkyCoord
        ICRF boresight.  Any adjustments due to pressure, temperature, relative
        humidity, or wavelength should be applied to the coord object.
    obstime : Time
        Time of observation
    location : EarthLocation
        Location of telescope

    Returns
    -------
    rsp : Angle
        rotSkyPos
    """
    q = pseudo_parallactic_angle(coord, obstime, location=location)
    return Angle(270*u.deg - rtp + q).wrap_at(180*u.deg)


def rsp_to_rtp(rsp, coord, obstime, location=RUBIN.location):
    """Compute rotSkyPos from rotTelPos

    Parameters
    ----------
    rsp : Angle
        Rotator position
    coord : SkyCoord
        ICRF boresight.  Any adjustments due to pressure, temperature, relative
        humidity, or wavelength should be applied to the coord object.
    obstime : Time
        Time of observation
    location : EarthLocation
        Location of telescope

    Returns
    -------
    rtp : Angle
        rotTelPos
    """
    q = pseudo_parallactic_angle(coord, obstime, location=location)
    return Angle(270*u.deg - rsp + q).wrap_at(180*u.deg)


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


def schedule_blocks(
    blocks,
    start_time,
    location=RUBIN.location,
    pressure=RUBIN.pressure,
    temperature=RUBIN.temperature,
    relative_humidity=RUBIN.relative_humidity,
    obswl=0.7*u.um
):
    """
    Schedule blocks

    Parameters
    ----------
    blocks : dict
        Dictionary of blocks
    start_time : Time
        Start time
    location : EarthLocation
        Location of telescope
    obswl : Quantity
        Wavelength of observation

    Returns
    -------
    schedule : QTable
        Schedule table
    """
    schedule = QTable(Block.block_schema)
    for block in blocks.values():
        schedule = vstack([schedule, block._table])
    n = len(schedule)

    for name in ['ra', 'dec', 'rsp', 'rtp', 'el', 'az']:
        if name not in schedule.colnames:
            schedule[name] = Masked(np.full(n, np.nan)*u.deg, mask=True)

    # First fill in obstime
    schedule['obstime'] = start_time + np.cumsum(schedule['elaptime'])
    schedule['obstime'].format='isot'

    # Where ra/dec are missing, compute from az/el
    w = schedule['ra'].mask & ~schedule['el'].mask
    if np.any(w):
        coord = SkyCoord(
            schedule['az'][w],
            schedule['el'][w],
            frame=AltAz(
                obstime=schedule['obstime'][w],
                location=location,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                obswl=obswl
            ),
        )
        schedule['ra'][w] = coord.icrs.ra
        schedule['dec'][w] = coord.icrs.dec

    # Where ra/dec/rsp are still missing, assume it's the same as previous
    # (i.e., we're tracking)
    prev_ra = None
    prev_dec = None
    for row in schedule:
        if not row['ra'].mask:
            prev_ra = row['ra']
            prev_dec = row['dec']
        else:
            row['ra'] = prev_ra
            row['dec'] = prev_dec

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
                location=location,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                obswl=obswl
            ),
            obstime=schedule['obstime'][w]
        )

    # Where rsp is still missing, assume it's the same as previous
    # (i.e., we're tracking)
    prev_rsp = None
    for row in schedule:
        if not row['rsp'].mask:
            prev_rsp = row['rsp']
        else:
            row['rsp'] = prev_rsp

    # We should now be able to go back and fill in missing az/el/rtp
    w = schedule['el'].mask & ~schedule['ra'].mask
    if np.any(w):
        coord = SkyCoord(
            schedule['ra'][w],
            schedule['dec'][w],
            obstime=schedule['obstime'][w],
            location=location,
            pressure=pressure,
            temperature=temperature,
            relative_humidity=relative_humidity,
            obswl=obswl
        )
        schedule['el'][w] = coord.altaz.alt
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
                location=location,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                obswl=obswl
            ),
            obstime=schedule['obstime'][w]
        )

    # Finally, add hour angle, pseudo-parallactic angle
    coord = SkyCoord(
        schedule['ra'], schedule['dec'],
        obstime=schedule['obstime'],
        location=location,
        pressure=pressure,
        temperature=temperature,
        relative_humidity=relative_humidity,
        obswl=obswl
    )
    schedule['ha'] = coord.hadec.ha

    schedule['q'] = pseudo_parallactic_angle(
        coord,
        obstime=schedule['obstime'],
        location=location
    ).to(u.deg)

    # Add a few columns
    schedule['dayobs'] = [int(o.datetime.strftime('%Y%m%d')) for o in schedule['obstime']-12*u.h]
    schedule['seqnum'] = np.empty(len(schedule), dtype=int)
    schedule['nightfrac'] = np.zeros(len(schedule))
    for d in np.unique(schedule['dayobs']):
        w = schedule['dayobs'] == d
        schedule['seqnum'][w] = np.arange(1, np.sum(w)+1)
        dstr = str(d)
        dstr = f"{dstr[:4]}-{dstr[4:6]}-{dstr[6:]}"
        night = almanac(dstr)
        t0 = night['sunset'][-12]
        t1 = night['sunrise'][-12]
        schedule['nightfrac'][w] = (schedule['obstime'][w]-t0)/(t1-t0)
    schedule['mjd'] = schedule['obstime'].tai.mjd

    # Reorder columns
    cols = schedule.colnames
    new_cols = ['dayobs', 'seqnum', 'nightfrac', 'block_name', 'band', 'mjd', 'ra', 'dec', 'rsp', 'az', 'el', 'rtp', 'ha', 'q']
    for col in cols:
        if col not in new_cols:
            new_cols.append(col)
    schedule = schedule[new_cols]
    return schedule


class PeekSky:
    def __init__(self):
        self.gaia = QTable.read("density.ecsv")
        n_points = len(self.gaia)
        self.n_side = int(np.sqrt(n_points / 12))
        self.indices = np.arange(hp.nside2npix(self.n_side))
        self.gaia_ra, self.gaia_dec = hp.pix2ang(
            self.n_side, self.indices, lonlat=True
        )

    @staticmethod
    def azel_to_postel(az, el):
        r = 90-el
        y = np.cos(np.deg2rad(az))*r
        x = -np.sin(np.deg2rad(az))*r
        return x, y

    def peek_sky(self, time, location=RUBIN.location, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='k')
            ax.set_facecolor('k')
        else:
            fig = ax.figure

        frame = AltAz(obstime=time, location=location)

        # Draw azel grid
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
        sunCoords.location = location
        if sunCoords.altaz.alt.value > 0:
            ax.text(
                *self.azel_to_postel(
                    sunCoords.altaz.az.deg,
                    sunCoords.altaz.alt.deg
                ),
                "☉",
                c='y',
                fontsize=25, weight='bold'
            )

        # Draw moon
        moonCoords = get_body('moon', time)
        moonCoords.location = location

        if moonCoords.altaz.alt.value > 0:
            ax.text(
                *self.azel_to_postel(
                    moonCoords.altaz.az.deg,
                    moonCoords.altaz.alt.deg
                ),
                "☽︎",
                c='c',
                fontsize=25, weight='bold'
            )

        # Draw Jupiter and Venus
        jupiterCoords = get_body('jupiter', time)
        jupiterCoords.location = location
        if jupiterCoords.altaz.alt.value > 0:
            ax.text(
                *self.azel_to_postel(
                    jupiterCoords.altaz.az.deg,
                    jupiterCoords.altaz.alt.deg
                ),
                "♃",
                c='r',
                fontsize=15, weight='bold'
            )

        venusCoords = get_body('venus', time)
        venusCoords.location = location
        if venusCoords.altaz.alt.value > 0:
            ax.text(
                *self.azel_to_postel(
                    venusCoords.altaz.az.deg,
                    venusCoords.altaz.alt.deg
                ),
                "♀",
                c='g',
                fontsize=15, weight='bold'
            )

        # Cardinal directions
        for az, el, label, color in [
            (0, 0, "N", 'r'),
            (90, 0, "E", 'r'),
            (180, 0, "S", 'r'),
            (270, 0, "W", 'r'),
            (45, 0, "45", 'g'),
            (135, 0, "135", 'g'),
            (225, 0, "225", 'g'),
            (315, 0, "315", 'g')
        ]:
            ax.text(
                *self.azel_to_postel(az, el),
                label,
                c=color,
                fontsize=15,
                verticalalignment='center',
                horizontalalignment='center'
            )
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        xs = np.linspace(-100, 100, 700)
        xs, ys = np.meshgrid(xs, xs)
        el = np.deg2rad(90 - np.hypot(xs, ys))
        w = el >= 0
        az = np.arctan2(-xs, ys)
        coords = SkyCoord(az*u.rad, el*u.rad, frame=frame)
        ra = coords.icrs.ra.deg
        dec = coords.icrs.dec.deg
        idxs = hp.ang2pix(self.n_side, ra, dec, lonlat=True)

        density = self.gaia[idxs]['density']
        density[~w] = np.nan

        ax.imshow(density, extent=[-100, 100, -100, 100], origin='lower')

        def xystr(x, y):
            az = np.rad2deg(np.arctan2(-x, y))%360
            el = 90 - np.hypot(x,y)
            c = SkyCoord(az*u.deg, el*u.deg, frame=frame)
            α = c.icrs.ra.deg
            δ = c.icrs.dec.deg
            return f"{az=:.1f} {el=:.1f} {α=:.1f} {δ=:.1f}"

        def hover(event):
            if event.inaxes == ax:
                ax.format_coord = xystr

        fig.canvas.mpl_connect('motion_notify_event', hover)
        fig.tight_layout()

        return fig, ax


def focus_sweep(
    az=None,
    el=None,
    rtp=None,
    axis=None,
    window=None,
    nstep=9,
    name=None
):
    if any(x is not None for x in [az, el, rtp]):
        if not all(x is not None for x in [az, el, rtp]):
            raise ValueError("If any of az, el, rtp are set, all must be set")
        mask = [False]+[True]*(nstep-1)
    else:
        az = np.nan*u.deg
        el = np.nan*u.deg
        rtp = np.nan*u.deg
        mask = [True]*nstep

    az = Masked(np.full(nstep, az.value)*az.unit, mask=mask)
    el = Masked(np.full(nstep, el.value)*el.unit, mask=mask)
    rtp = Masked(np.full(nstep, rtp.value)*rtp.unit, mask=mask)

    if"_d" in axis:
        if "_shift" not in axis:
            axis = axis.replace("_d", "_shift_d")
    if "_r" in axis:
        if "_rot" not in axis:
            axis = axis.replace("_r", "_rot_r")

    if window is None:
        match axis:
            case 'cam_shift_dz' | 'm2_shift_dz':
                window = 300*u.micron
            case 'cam_shift_dx' | 'cam_shift_dy':
                window = 14000*u.micron
            case 'cam_rot_rx' | 'cam_rot_ry':
                window = 900*u.arcsec
            case 'm2_shift_dx' | 'm2_shift_dy':
                window = 7000*u.micron
            case 'm2_rot_rx' | 'm2_rot_ry':
                window = 240*u.arcsec
            case _:
                raise ValueError(f"Unknown axis {axis}")
    vals = np.linspace(-window/2, window/2, nstep)
    if name is None:
        name = f"sweep_{axis}".replace("_shift", "").replace("_rot", "")
    block = Block(
        az=az,
        el=el,
        rtp=rtp,
        **{axis: vals},
        n=nstep,
        name=name
    )
    return block


def sensitivity(
    az,
    el,
    axis=None,
    window=None,
    ntriplet=9,
    name=None
):
    mask = np.tile([False, True, True], ntriplet)  # Ratchet back every 3rd exposure, track in between.
    # Rotate from -85 to 85 degrees in rtp.
    rtp = np.full(3*ntriplet, np.nan)*u.deg
    rtp[::3] = np.linspace(-85, 85, ntriplet)*u.deg
    rtp = Masked(rtp, mask=mask)
    az = Masked(np.tile([az.value, np.nan, np.nan], ntriplet)*az.unit, mask=mask)
    el = Masked(np.tile([el.value, np.nan, np.nan], ntriplet)*el.unit, mask=mask)

    if"_d" in axis:
        if "_shift" not in axis:
            axis = axis.replace("_d", "_shift_d")
    if "_r" in axis:
        if "_rot" not in axis:
            axis = axis.replace("_r", "_rot_r")

    if window is None:
        if (match := re.match(r"^(cam|m2)_shift_dz$", axis)):
            window = 300*u.micron
        elif (match := re.match(r"^(cam|m2)_shift_(dx|dy)$", axis)):
            if match.group(1) == "cam":
                window = 14000*u.micron
            else:
                window = 7000*u.micron
        elif (match := re.match(r"^(cam|m2)_rot_(rx|ry)$", axis)):
            if match.group(1) == "cam":
                window = 900*u.arcsec
            else:
                window = 240*u.arcsec
        elif (match := re.match(r"^m1m3_b(\d+)$", axis)):
            window = 2*u.micron
        elif (match := re.match(r"^m2_b(\d+)$", axis)):
            window = 1*u.micron
        else:
            raise ValueError(f"Unknown axis {axis}")
    focusZ = np.tile([-1.5, 1.5, 0.0]*u.mm, ntriplet)  # intra/extra/focal
    vals = np.repeat(np.linspace(-window/2, window/2, ntriplet), 3)
    if name is None:
        name = f"sens_{axis}".replace("_shift", "").replace("_rot", "")
    block = Block(
        az=az,
        el=el,
        rtp=rtp,
        focusZ=focusZ,
        **{axis: vals},
        n=3*ntriplet,
        name=name
    )
    return block


def reference(
    az=None,
    el=None,
    rtp=None,
    focusZ=0*u.mm,
    n=10,
    name="reference"
):
    if any(x is not None for x in [az, el, rtp]):
        if not all(x is not None for x in [az, el, rtp]):
            raise ValueError("If any of az, el, rtp are set, all must be set")
        mask = [False]+[True]*(n-1)
    else:
        az = np.nan*u.deg
        el = np.nan*u.deg
        rtp = np.nan*u.deg
        mask = [True]*n

    az = Masked(np.full(n, az.value)*az.unit, mask=mask)
    el = Masked(np.full(n, el.value)*el.unit, mask=mask)
    rtp = Masked(np.full(n, rtp.value)*rtp.unit, mask=mask)

    block = Block(
        az=az,
        el=el,
        rtp=rtp,
        focusZ=focusZ,
        n=n,
        name=name
    )
    return block
