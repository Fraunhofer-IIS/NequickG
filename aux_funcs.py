import numpy as np


def coord2cartesian(r, lat, lon):
    """
    Converts from geo to cartesian coordinates
    :param r: [km]
    :param lat: [deg]
    :param lon: [deg]
    :return:
    """
    dr = np.pi / 180
    x = r * np.cos(lat * dr) * np.cos(lon * dr)
    y = r * np.cos(lat * dr) * np.sin(lon * dr)
    z = r * np.sin(lat * dr)

    return x, y, z


def cartesian2coord(x, y, z):
    """
    Converts from cartesian to geo coordinates
    :param x: [km]
    :param y: [km]
    :param z: [km]
    :return:
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    xy = np.sqrt(x ** 2 + y ** 2)

    lat = np.arctan(z / xy) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi

    return r, lat, lon


def interpolate(z1, z2, z3, z4, x):
    """
    Third Order Lagrange Interpolation function
    Reference: Section 2.5.7.1 of GSA's "Ionospheric Correction
    Algorithm for Galileo Single Frequency Users"

    """
    # if abs(2 * x) < 10 ** -10:
    #     return z2

    delta = 2 * x - 1
    g1 = z3 + z2
    g2 = z3 - z2
    g3 = z4 + z1
    g4 = (z4 - z1) / 3.0

    a0 = 9 * g1 - g3
    a1 = 9 * g2 - g4
    a2 = g3 - g1
    a3 = g4 - g2

    return 1 / 16.0 * (a0 + a1 * delta + a2 * delta ** 2 + a3 * delta ** 3)


def interpolate2d(z, x, y):
    assert (np.shape(z) == (4, 4))

    deltax = 2 * x - 1
    deltay = 2 * y - 1
    # Interpolate horizontally first

    g1 = z[2, :] + z[1, :]
    g2 = z[2, :] - z[1, :]
    g3 = z[3, :] + z[0, :]
    g4 = (z[3, :] - z[0, :]) / 3.0

    a0 = 9 * g1 - g3
    a1 = 9 * g2 - g4
    a2 = g3 - g1
    a3 = g4 - g2

    z = 1 / 16.0 * (a0 + a1 * deltay + a2 * deltay ** 2 + a3 * deltay ** 3)

    g1 = z[2] + z[1]
    g2 = z[2] - z[1]
    g3 = z[3] + z[0]
    g4 = (z[3] - z[0]) / 3.0

    a0 = 9 * g1 - g3
    a1 = 9 * g2 - g4
    a2 = g3 - g1
    a3 = g4 - g2

    return 1 / 16.0 * (a0 + a1 * deltax + a2 * deltax ** 2 + a3 * deltax ** 3)


def epstein(peak_amp, peak_height, thickness, h):
    return peak_amp * neq_clip_exp((h - peak_height) / thickness) / np.power((1 + neq_clip_exp((h - peak_height) / thickness)), 2)


def neq_join(d_f1, d_f2, d_alpha, d_x):
    """
    Allows smooth joining of functions f1 and f2 (i.e. continuous first derivatives) at origin.
    Alpha determines width of transition region. Calculates value of joined functions at x.
    :param d_f1:
    :param d_f2:
    :param d_alpha:
    :param d_x:
    :return:
    """
    ee = neq_clip_exp(d_alpha * d_x)
    return (d_f1 * ee + d_f2) / (ee + 1)


def neq_clip_exp(d_power):
    """

    :param d_power: Power for exponential function [double]
    :return:
    """

    assert (not np.any(np.isnan(d_power)))

    mask1 = np.logical_and(d_power < 80, d_power > -80)
    mask2 = d_power > 80
    mask3 = d_power < -80
    out = np.exp(d_power, where=mask1)
    if type(out) == np.ndarray:
        out[mask2] = 5.5406 * 10 ** 34
        out[mask3] = 1.8049 * 10 ** -35
    else:
        if mask2:
            out = 5.5406 * 10 ** 34
        elif mask3:
            out = 1.8049 * 10 ** -35

    assert (not np.any(out < 0))
    return out


def neq_critical_freq_to_ne(f0):
    """
    :param f0: peak plasma frequency of layer [MHz]
    :return:
    """
    return 0.124 * f0 ** 2


if __name__ == "__main__":
    # unit testing
    assert neq_clip_exp(-100) == 1.8049 * 10 ** -35
    assert neq_clip_exp(100) == 5.5406 * 10 ** 34

    assert np.all(neq_clip_exp(np.array([-100, 0, 100])) == np.array([1.8049 * 10 ** -35, 1., 5.5406 * 10 ** 34]))
