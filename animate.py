import astropy.units as u
import healpy as hp
import matplotlib.animation as animation
import numpy as np
from astropy.coordinates import AltAz, SkyCoord, get_body
from astropy.table import QTable
from astropy.time import Time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from aos_scheduler import RUBIN


class ScheduleAnimation:
    def __init__(self, schedule, axis=None, location=RUBIN.location):
        self.schedule = schedule
        self.nframe = len(schedule)
        self.iframe = 0

        if axis is None:
            fig = Figure(facecolor="k")
            axis = fig.add_axes([0, 0, 1, 1])
        self.axis = axis
        self.axis.set_aspect("equal")
        self.fig = self.axis.get_figure()
        self.axis.set_facecolor("k")
        self.axis.set_xlim(-100, 100)
        self.axis.set_ylim(-100, 100)

        self.location = location

        self.gaia = QTable.read("density.ecsv")
        n_points = len(self.gaia)
        self.n_side = int(np.sqrt(n_points / 12))
        self.indices = np.arange(hp.nside2npix(self.n_side))
        self.gaia_ra, self.gaia_dec = hp.pix2ang(self.n_side, self.indices, lonlat=True)

        self.draw_altaz_grid()
        self.draw_cardinal_points()
        self.init_radec_grid()
        self.init_bodies()
        self.init_gaia()
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.tight_layout()
        self.init_obs_arrow()
        self.init_texts()
        self.set_iframe(0)

    @staticmethod
    def azalt_to_postel(az, alt):
        r = 90 - alt
        y = np.cos(np.deg2rad(az)) * r
        x = -np.sin(np.deg2rad(az)) * r
        return x, y

    def draw_altaz_grid(self):
        az = np.linspace(0, 2 * np.pi, 100)
        for zen in np.arange(0, 90 + 1, 15):
            y, x = zen * np.cos(az), -zen * np.sin(az)
            self.axis.plot(x, y, c="g", lw=0.4)
        zen = np.linspace(0, 90)
        for az in np.arange(0, 2 * np.pi, np.pi / 4):
            y, x = zen * np.cos(az), -zen * np.sin(az)
            self.axis.plot(x, y, c="g", lw=(1.0 if az == 0 else 0.4))

    def init_radec_grid(self):
        self.radec_grid = self.axis.plot(
            [np.nan] * 2020, [np.nan] * 2020, c="b", lw=0.4
        )

    def calc_radec_grid(self):
        x = []
        y = []
        ra = np.linspace(0, 2 * np.pi, 100)
        for dec in np.arange(-90, 90, 15):
            c = SkyCoord(ra * u.rad, dec * u.deg)
            c = c.transform_to(self.frame)
            zen = 90 - c.altaz.alt.deg
            zen[zen > 90] = np.nan
            y1, x1 = zen * np.cos(c.altaz.az), -zen * np.sin(c.altaz.az)
            x.append(x1)
            y.append(y1)
            x.append([np.nan])
            y.append([np.nan])
        dec = np.linspace(-90, 90, 100)
        for ra in np.arange(0, 2 * np.pi, np.pi / 4):
            c = SkyCoord(ra * u.rad, dec * u.deg)
            c = c.transform_to(self.frame)
            zen = 90 - c.altaz.alt.deg
            zen[zen > 90] = np.nan
            y1, x1 = zen * np.cos(c.altaz.az), -zen * np.sin(c.altaz.az)
            x.append(x1)
            y.append(y1)
            x.append([np.nan])
            y.append([np.nan])
        x = np.concatenate(x)
        y = np.concatenate(y)
        return x, y

    def update_radec_grid(self):
        x, y = self.calc_radec_grid()
        self.radec_grid[0].set_data(x, y)

    def init_bodies(self):
        self.bodies = {}
        for name, symbol, color in zip(
            ["sun", "moon", "venus", "jupiter"],
            ["☉", "☽︎", "♀", "♃"],
            ["y", "c", "g", "r"],
        ):
            self.bodies[name] = [
                self.axis.text(
                    0.0,
                    0.0,
                    symbol,
                    c=color,
                    fontsize=25,
                    weight="bold",
                    ha="center",
                    va="center",
                ),
                symbol,
            ]

    def update_bodies(self):
        for name in self.bodies.keys():
            body = get_body(name, self.time)
            body.location = self.location
            if body.altaz.alt.deg > 0:
                self.bodies[name][0].set_text(self.bodies[name][1])
                x, y = self.azalt_to_postel(body.altaz.az.deg, body.altaz.alt.deg)
                self.bodies[name][0].set_x(x)
                self.bodies[name][0].set_y(y)
            else:
                self.bodies[name][0].set_text(" ")

    def draw_cardinal_points(self):
        for az, label, color in [
            (0, "N", "r"),
            (90, "E", "r"),
            (180, "S", "r"),
            (270, "W", "r"),
            (45, "45", "g"),
            (135, "135", "g"),
            (225, "225", "g"),
            (315, "315", "g"),
        ]:
            self.axis.text(
                *self.azalt_to_postel(az, -10),
                label,
                c=color,
                fontsize=15,
                verticalalignment="center",
                horizontalalignment="center",
            )

    def set_time(self, time, **kwargs):
        if not isinstance(time, Time):
            time = Time(time, **kwargs)
        self.time = time
        self.frame = AltAz(obstime=self.time, location=self.location)
        self.update_radec_grid()
        self.update_bodies()
        self.update_gaia()
        self.update_texts()

    def set_iframe(self, iframe):
        self.iframe = iframe
        time = self.schedule[iframe]["obstime"]
        self.set_time(time)
        self.update_obs_arrow()

    def init_gaia(self):
        self.gaia_im = self.axis.imshow(
            np.zeros((700, 700)),
            extent=[-100, 100, -100, 100],
            vmin=0,
            vmax=1.2e6,
            origin="lower",
        )

    def update_gaia(self):
        xs = np.linspace(-100, 100, 700)
        xs, ys = np.meshgrid(xs, xs)
        alt = np.deg2rad(90 - np.hypot(xs, ys))
        w = alt >= 0
        az = np.arctan2(-xs, ys)
        coords = SkyCoord(az * u.rad, alt * u.rad, frame=self.frame)
        ra = coords.icrs.ra.deg
        dec = coords.icrs.dec.deg
        idxs = hp.ang2pix(self.n_side, ra, dec, lonlat=True)

        density = self.gaia[idxs]["density"].value
        density[~w] = np.nan

        self.gaia_im.set_array(density)

    def hover(self, event):
        def xystr(x, y):
            alt = 90 - np.hypot(x, y)
            az = np.rad2deg(np.arctan2(-x, y)) % 360
            c = SkyCoord(az * u.deg, alt * u.deg, frame=self.frame)
            α = c.icrs.ra.deg
            δ = c.icrs.dec.deg
            return f"{alt=:.1f} {az=:.1f} {α=:.1f} {δ=:.1f}"

        if event.inaxes == self.axis:
            self.axis.format_coord = xystr

    def init_obs_arrow(self):
        self.obs_arrow = self.axis.arrow(
            0.0,
            0.0,
            0.0,
            0.0,
            color="red",
            width=0.1,
            head_width=2,
            length_includes_head=True,
            zorder=10,
        )

    def update_obs_arrow(self):
        self.obs_arrow.remove()
        row = self.schedule[self.iframe]
        focusz = float(row["focusZ"].to_value(u.mm))
        az, alt = float(row["az"].to_value(u.deg)), float(row["alt"].to_value(u.deg))
        x, y = self.azalt_to_postel(az, alt)

        # +Y_CCS
        th = np.arctan2(y, x) - float(row["rtp"].to_value(u.rad))
        dx = (10 + 2 * focusz) * np.cos(th)
        dy = (10 + 2 * focusz) * np.sin(th)
        x0 = x - dx / 2
        y0 = y - dy / 2

        self.obs_arrow = self.axis.arrow(
            x0,
            y0,
            dx,
            dy,
            color="red",
            width=0.1,
            head_width=2,
            length_includes_head=True,
            zorder=10,
        )

    def init_texts(self):
        kwargs = dict(
            color="w",
            fontsize=10,
            transform=self.fig.transFigure,
            fontdict={"family": "monospace", "weight": "bold"},
        )
        self.texts = {}
        self.texts["name"] = self.axis.text(0.02, 0.95, f'NAME: {"":20s}', **kwargs)
        self.texts["dayobs"] = self.axis.text(0.02, 0.92, f"DAYOBS: {0:08d}", **kwargs)
        self.texts["seqnum"] = self.axis.text(0.02, 0.89, f"SEQNUM: {0:04d}", **kwargs)
        self.texts["mjd"] = self.axis.text(0.02, 0.86, f"MJD:  {0: 9.4f}", **kwargs)
        self.texts["band"] = self.axis.text(0.02, 0.83, f'BAND: {"":5s}', **kwargs)
        self.texts["ra"] = self.axis.text(0.78, 0.95, f"RA:  {0: 9.4f}", **kwargs)
        self.texts["dec"] = self.axis.text(0.78, 0.92, f"DEC: {0: 9.4f}", **kwargs)
        self.texts["rsp"] = self.axis.text(0.78, 0.89, f"RSP: {0: 9.4f}", **kwargs)
        self.texts["az"] = self.axis.text(0.78, 0.86, f"AZ:  {0: 9.4f}", **kwargs)
        self.texts["alt"] = self.axis.text(0.78, 0.83, f"ALT: {0: 9.4f}", **kwargs)
        self.texts["rtp"] = self.axis.text(0.78, 0.80, f"RTP: {0: 9.4f}", **kwargs)
        self.texts["q"] = self.axis.text(0.78, 0.77, f"Q: {0: 9.4f}", **kwargs)
        self.texts["focusz"] = self.axis.text(
            0.02, 0.2, f"FOCUSZ: {0: 5.2f} mm", **kwargs
        )
        self.texts["cam_dz"] = self.axis.text(
            0.02, 0.17, f"CAM_DZ: {0: 5.2f} mm", **kwargs
        )
        self.texts["cam_dx"] = self.axis.text(
            0.02, 0.14, f"CAM_DX: {0: 5.2f} mm", **kwargs
        )
        self.texts["cam_dy"] = self.axis.text(
            0.02, 0.11, f"CAM_DY: {0: 5.2f} mm", **kwargs
        )
        self.texts["cam_rx"] = self.axis.text(
            0.02, 0.08, f"CAM_RX: {0: 5.2f} arcsec", **kwargs
        )
        self.texts["cam_ry"] = self.axis.text(
            0.02, 0.05, f"CAM_RY: {0: 5.2f} arcsec", **kwargs
        )
        self.texts["m2_dz"] = self.axis.text(
            0.78, 0.17, f"M2_DZ: {0: 5.2f} mm", **kwargs
        )
        self.texts["m2_dx"] = self.axis.text(
            0.78, 0.14, f"M2_DX: {0: 5.2f} mm", **kwargs
        )
        self.texts["m2_dy"] = self.axis.text(
            0.78, 0.11, f"M2_DY: {0: 5.2f} mm", **kwargs
        )
        self.texts["m2_rx"] = self.axis.text(
            0.78, 0.08, f"M2_RX: {0: 5.2f} arcsec", **kwargs
        )
        self.texts["m2_ry"] = self.axis.text(
            0.78, 0.05, f"M2_RY: {0: 5.2f} arcsec", **kwargs
        )

    def update_texts(self):
        row = self.schedule[self.iframe]
        self.texts["name"].set_text(f'NAME: {row["block_name"]:20s}')
        self.texts["dayobs"].set_text(f'DAYOBS: {row["dayobs"]:08d}')
        self.texts["seqnum"].set_text(f'SEQNUM: {row["seqnum"]:04d}')
        self.texts["mjd"].set_text(f'MJD:  {row["mjd"]: 9.4f}')
        self.texts["band"].set_text(f'BAND: {row["band"]:5s}')
        self.texts["ra"].set_text(f'RA:  {row["ra"]: 9.4f}')
        self.texts["dec"].set_text(f'DEC: {row["dec"]: 9.4f}')
        self.texts["rsp"].set_text(f'RSP: {row["rsp"]: 9.4f}')
        self.texts["az"].set_text(f'AZ:  {row["az"]: 9.4f}')
        self.texts["alt"].set_text(f'ALT: {row["alt"]: 9.4f}')
        self.texts["rtp"].set_text(f'RTP: {row["rtp"]: 9.4f}')
        self.texts["q"].set_text(f'Q: {row["q"]: 9.4f}')
        self.texts["focusz"].set_text(
            f'FOCUSZ: {row["focusZ"].to_value(u.mm): 5.2f} mm'
        )
        self.texts["cam_dz"].set_text(
            f'CAM_DZ: {row["cam_shift"]["dz"].to_value(u.mm): 5.2f} mm'
        )
        self.texts["cam_dx"].set_text(
            f'CAM_DX: {row["cam_shift"]["dx"].to_value(u.mm): 5.2f} mm'
        )
        self.texts["cam_dy"].set_text(
            f'CAM_DY: {row["cam_shift"]["dy"].to_value(u.mm): 5.2f} mm'
        )
        self.texts["cam_rx"].set_text(
            f'CAM_RX: {row["cam_rot"]["rx"].to_value(u.arcsec): 5.2f} arcsec'
        )
        self.texts["cam_ry"].set_text(
            f'CAM_RY: {row["cam_rot"]["ry"].to_value(u.arcsec): 5.2f} arcsec'
        )
        self.texts["m2_dz"].set_text(
            f'M2_DZ: {row["m2_shift"]["dz"].to_value(u.mm): 5.2f} mm'
        )
        self.texts["m2_dx"].set_text(
            f'M2_DX: {row["m2_shift"]["dx"].to_value(u.mm): 5.2f} mm'
        )
        self.texts["m2_dy"].set_text(
            f'M2_DY: {row["m2_shift"]["dy"].to_value(u.mm): 5.2f} mm'
        )
        self.texts["m2_rx"].set_text(
            f'M2_RX: {row["m2_rot"]["rx"].to_value(u.arcsec): 5.2f} arcsec'
        )
        self.texts["m2_ry"].set_text(
            f'M2_RY: {row["m2_rot"]["ry"].to_value(u.arcsec): 5.2f} arcsec'
        )
