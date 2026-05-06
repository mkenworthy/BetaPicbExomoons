import numpy as np
from astropy import constants as c
from astropy import units as u

# https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract

def chen_mass_radius(mass):
    ' from Chen and Kipping 2017 ApJ 834:17 Figure 3'
    radius = np.zeros(mass.shape)*u.Rearth

    m1 = 2.04*u.Mearth
    m2 = 0.414*u.Mjup
    m3 = 0.080*u.Msun

    r1 = 1.23 * u.Rearth
    r2 = 1.5e1 * u.Rearth
    r3 = 1.1e1 * u.Rearth

    p1 = 0.279

    p3 = -0.04

    A1 = (r1/u.Rearth)/np.power(m1/u.Mearth,p1)
    A2 = (r2/u.Rearth)/np.power(m2/u.Mearth,0.59)
    A3 = (r3/u.Rearth)/np.power(m3/u.Mearth,p3)
    A4 = (r3/u.Rearth)/np.power(m3/u.Mearth,0.88)

    terran = (mass <= m1)
    neptun = (mass>m1) * (mass <= m2)
    jovian = (mass>m2) * (mass <= m3)
    stellar = (mass>m3)

    radius[terran] = A1 * np.power(mass[terran]/u.Mearth,p1)*u.Rearth
    radius[neptun] = A2 * np.power(mass[neptun]/u.Mearth,0.59)*u.Rearth
    radius[jovian] = A3 * np.power(mass[jovian]/u.Mearth,p3)*u.Rearth
    radius[stellar] = A4 * np.power(mass[stellar]/u.Mearth,0.88)*u.Rearth

    return radius.to(u.Rearth)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    x = np.logspace(-3.5,6)*u.Mearth
    rad = chen_mass_radius(x)
    ax.loglog(x, rad.value,'o--')
    ax.grid()
    ax.grid(which='minor', color="0.9")
    ax.set_ylabel('Radius [REarth]')
    ax.set_xlabel('Mass [Mearth]')
    plt.draw()
    plt.show()
