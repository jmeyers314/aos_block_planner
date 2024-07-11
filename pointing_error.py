import numpy as np
from aos_scheduler import rsp_to_rtp, rtp_to_rsp, RUBIN
from astropy.coordinates import AltAz, ICRS, Angle, SkyCoord
from astropy.time import Time
import astropy.units as u


def track(alt, az, rtp, exptime):
    time = Time("J2024")  # arbitrary

    aa0 = SkyCoord(az, alt, frame=AltAz, obstime=time, location=RUBIN.location)
    rsp0 = rtp_to_rsp(rtp, aa0.icrs, time)
    icrs1 = SkyCoord(
        aa0.icrs.ra, aa0.icrs.dec,
        frame=ICRS, obstime=time+exptime,
        location=RUBIN.location
    )
    rtp1 = rsp_to_rtp(rsp0, icrs1, time+exptime)
    return icrs1.altaz.alt, icrs1.altaz.az, rtp1


def main(args):
    # Start with the telescope pointing
    # at (alt, az, rtp=0).  Calculate where
    # the telescope will point after exptime.
    # That gives dalt1, daz1, drtp1.
    # Then calculate where a hypothetical telescope
    # pointing at (alt+dalt, az+daz, rtp=0) will point
    # after exptime.  That gives dalt2, daz2, drtp2.
    # The tracking error is then dalt2-dalt1, daz2-daz1, drtp2-drtp1.
    # Finally, do some additional math to estimate the PSF
    # impact in arcseconds both on-axis and at 0.5 deg off-axis.

    alt1, az1, rtp1 = track(
        args.alt*u.deg, args.az*u.deg, 0.0*u.deg, args.exptime*u.s
    )
    alt2, az2, rtp2 = track(
        (args.alt+args.dalt)*u.deg, (args.az+args.daz)*u.deg, 0.0*u.deg, args.exptime*u.s
    )

    dalt1 = Angle(alt1-args.alt*u.deg).wrap_at(180*u.deg).deg
    daz1 = Angle(az1-args.az*u.deg).wrap_at(180*u.deg).deg
    drtp1 = Angle(rtp1-0.0*u.deg).wrap_at(180*u.deg).deg
    dalt2 = Angle(alt2-(args.alt+args.dalt)*u.deg).wrap_at(180*u.deg).deg
    daz2 = Angle(az2-(args.az+args.daz)*u.deg).wrap_at(180*u.deg).deg
    drtp2 = Angle(rtp2-0.0*u.deg).wrap_at(180*u.deg).deg

    print()
    print("True tracking")
    print("dalt: {:.6f} deg".format(dalt1))
    print("daz:  {:.6f} deg".format(daz1))
    print("drtp: {:.6f} deg".format(drtp1))
    print()
    print("Attempted tracking")
    print("dalt: {:.6f} deg".format(dalt2))
    print("daz:  {:.6f} deg".format(daz2))
    print("drtp: {:.6f} deg".format(drtp2))
    print()
    print("PSF impact")
    print("dalt: {:.6f} arcsec".format(np.abs(dalt2-dalt1)*3600/np.sqrt(12)))
    print("daz:  {:.6f} arcsec".format(np.cos(alt1)*np.abs(daz2-daz1)*3600/np.sqrt(12)))
    print("drtp: {:.6f} arcsec".format(np.abs(np.deg2rad(drtp2-drtp1))*0.5*3600/np.sqrt(12)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pointing error calculation')
    parser.add_argument('alt', type=float, help='Altitude (deg)')
    parser.add_argument('az', type=float, help='Azimuth (deg)')
    parser.add_argument('dalt', type=float, help='Altitude offset (deg)')
    parser.add_argument('daz', type=float, help='Azimuth offset (deg)')
    parser.add_argument('--exptime', type=float, default=30.0, help='Exposure time (s)')
    args = parser.parse_args()

    main(args)
