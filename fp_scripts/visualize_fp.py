"""
This script provides the main function to handle visualization of the projected Primecam Module Focalplane using matplotlib.
It allows customization of the plot with various options such as labels, colors, and coordinate systems.
The base script is from the TOAST package, with modifications to allow for custom color and polarization arrow plotting.
"""

import numpy as np
import astropy.units as u
from toast import qarray as qa
from toast.instrument_coords import quat_to_xieta
from toast.vis import set_matplotlib_backend

def plot_focalplane_primecam(
    focalplane=None,
    width=None,
    height=None,
    outfile=None,
    show_labels=False,
    face_color=None,
    pol_color=None,
    xieta=False,
    show_centers=False,
    show_gamma=False,
):
    """Visualize a projected Primecam Module Focalplane. This is a modified version of the
    TOAST function plot_focalplane.

    This makes a simple plot of the detector positions on the projected focalplane.
    By default, this plots the focalplane in the boresight X / Y / Z frame, as seen
    by incoming photons.  If `xieta` is set to True, the focalplane is plotted in
    Xi / Eta / Gamma coordinates as seen from the observer looking out at the sky.

    To avoid python overhead in large MPI jobs, we place the matplotlib import inside
    this function, so that it is only imported when the function is actually called.

    Args:
        focalplane (Focalplane):  The focalplane to plot
        width (Quantity):  Width of plot.
        height (Quantity):  Height of plot.
        outfile (str):  Output PDF path.  If None, then matplotlib will be
            used for inline plotting.
        show_labels (bool):  If True, plot detector names.
        face_color (dict): Mapping from module names in the focalplane QTable's
            ``module`` column to detector face colors. Modules not present in
            the mapping are gray.
        pol_color (dict): dictionary of color values for the polarization
            arrows.
        xieta (bool):  Plot in observer xi/eta/gamma coordinates rather than
            boresight X/Y/Z.
        show_centers (bool):  If True, label the pixel centers.
        show_gamma (bool):  If True, show gamma angle (for debugging).

    Returns:
        (Figure):  The figure.

    """
    if focalplane is None:
        raise RuntimeError("You must specify a Focalplane instance")

    if outfile is not None:
        set_matplotlib_backend(backend="pdf")

    import matplotlib.pyplot as plt

    if width is None:
        width = 10.0 * u.degree

    if height is None:
        height = 10.0 * u.degree

    width_deg = width.to_value(u.degree)
    height_deg = height.to_value(u.degree)

    # xfigsize = int(width_deg) + 1
    # yfigsize = int(height_deg) + 1
    xfigsize = 6
    yfigsize = 6
    figdpi = 100

    # Compute the font size to use for detector labels
    fontpix = 0.05 * figdpi
    fontpt = int(0.75 * fontpix)

    fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
    # fig = plt.figure(figsize=(5, 5), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    half_width = 0.6 * width_deg
    half_height = 0.6 * height_deg
    if xieta:
        ax.set_xlabel(r"Boresight $\xi$ Degrees", fontsize="medium")
        ax.set_ylabel(r"Boresight $\eta$ Degrees", fontsize="medium")
    else:
        ax.set_xlabel("Boresight X Degrees", fontsize="medium")
        ax.set_ylabel("Boresight Y Degrees", fontsize="medium")
    ax.set_xlim([-half_width, half_width])
    ax.set_ylim([-half_height, half_height])

    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    yaxis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for d in focalplane.detectors:
        quat = focalplane[d]["quat"]
        fwhm = focalplane[d]["fwhm"].to_value(u.arcmin)

        # radius in degrees
        detradius = 0.5 * 5.0 / 60.0
        if fwhm is not None:
            detradius = 0.5 * fwhm / 60.0
            # detradius = 0.4 * fwhm / 60.0
            
        if xieta:
            xi, eta, gamma = quat_to_xieta(quat)
            xpos = xi * 180.0 / np.pi
            ypos = eta * 180.0 / np.pi
            # Polang is plotted relative to visualization x/y coords
            polang = 1.5 * np.pi - gamma
            plot_gamma = polang
        else:
            # rotation from boresight
            rdir = qa.rotate(quat, zaxis).flatten()
            mag = np.arccos(rdir[2]) * 180.0 / np.pi
            ang = np.arctan2(rdir[1], rdir[0])
            orient = qa.rotate(quat, xaxis).flatten()
            polang = np.arctan2(orient[1], orient[0])
            xpos = mag * np.cos(ang)
            ypos = mag * np.sin(ang)
            xi, eta, gamma = quat_to_xieta(quat)
            plot_gamma = gamma

        ascale = 0.7 #1.5

        xtail = xpos - ascale * detradius * np.cos(polang)
        ytail = ypos - ascale * detradius * np.sin(polang)
        dx = ascale * 2.0 * detradius * np.cos(polang)
        dy = ascale * 2.0 * detradius * np.sin(polang)

        if show_centers:
            ysgn = -1.0
            if dx < 0.0:
                ysgn = 1.0
            ax.text(
                (xpos + 0.1 * dx),
                (ypos + 0.1 * ysgn * dy),
                f"({xpos:0.4f}, {ypos:0.4f})",
                color="green",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
            )

        if show_labels:
            xsgn = 1.0
            if dx < 0.0:
                xsgn = -1.0
            labeloff = 0.05 * xsgn * fontpix * len(d) / figdpi
            ax.text(
                (xtail + 1.3 * dx + labeloff),
                (ytail + 1.2 * dy),
                d,
                color="k",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
            )

        if show_gamma:
            ax.arrow(
                xtail,
                ytail,
                1.3 * dx,
                1.3 * dy,
                width = 0.1 * detradius,
                head_width=0.2 * detradius,
                head_length=0.2 * detradius,
                fc="gray",
                ec="gray",
                length_includes_head=True,
            )

        # This part plots the polarisation arrows
        detcolor = "black"
        if pol_color is not None:
            detcolor = pol_color[d]
            ax.arrow(
            xtail,
            ytail,
            dx,
            dy,
            width= 0.01 *detradius,
            head_width= 0.3 *detradius,
            head_length= 0.3 *detradius,
            fc=detcolor,
            ec=detcolor,
            length_includes_head=False,
            alpha=0.5
            )

        detface = "gray"
        if face_color is not None:
            try:
                module_name = focalplane[d]["module"]
            except (KeyError, ValueError) as error:
                raise RuntimeError(
                    "face_color requires a 'module' column in the focalplane "
                    "QTable"
                ) from error
            detface = face_color.get(str(module_name), "gray")

        if pol_color is not None:
            circ = plt.Circle((xpos, ypos), radius=detradius, fc="none", 
                              ec="k", linewidth=0.3)

        else:
            circ = plt.Circle((xpos, ypos), radius=detradius, fc=detface)
        ax.add_artist(circ)
    
    # Draw a "mini" coordinate axes for reference
    # xmini = -0.8 * half_width
    # ymini = -0.8 * half_height
    # xlen = 0.1 * half_width
    # ylen = 0.1 * half_height
    # mini_width = 0.005 * half_width
    # mini_head_width = 3 * mini_width
    # mini_head_len = 3 * mini_width
    # if xieta:
    #     aprops = [
    #         (xlen, 0, "-", r"$\xi$"),
    #         (0, ylen, "-", r"$\eta$"),
    #         (-xlen, 0, "--", "Y"),
    #         (0, -ylen, "--", "X"),
    #     ]
    # else:
    #     aprops = [
    #         (xlen, 0, "-", "X"),
    #         (0, ylen, "-", "Y"),
    #         (-xlen, 0, "--", r"$\eta$"),
    #         (0, -ylen, "--", r"$\xi$"),
    #     ]
    # for ap in aprops:
    #     lx = xmini + 1.5 * ap[0]
    #     ly = ymini + 1.5 * ap[1]
    #     lw = figdpi / 200.0
    #     ax.arrow(
    #         xmini,
    #         ymini,
    #         ap[0],
    #         ap[1],
    #         width=mini_width,
    #         head_width=mini_head_width,
    #         head_length=mini_head_len,
    #         fc="k",
    #         ec="k",
    #         linestyle=ap[2],
    #         linewidth=lw,
    #         length_includes_head=True,
    #     )
    #     ax.text(
    #         lx,
    #         ly,
    #         ap[3],
    #         color="k",
    #         fontsize=int(figdpi / 10),
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #     )

    # st = "Focalplane Looking Towards Observer"
    # if xieta:
    #     st = "Focalplane on Sky From Observer"
    # fig.suptitle(st)

    if outfile is None:
        output_plt = plt.show();
        print()
    else:
        plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
        plt.close()
    return fig
