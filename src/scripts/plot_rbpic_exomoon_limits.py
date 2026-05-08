debug=1
import numpy as np
from astropy import constants as c
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import ascii
from chen_mass_radius import *
from betapic import *
from kepler3 import *
from matplotlib.patches import Polygon,Rectangle
import paths

if debug:
    import matplotlib
    matplotlib.use('MacOSX')

plt.rcParams.update({
#    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Helvetica",
})


@u.quantity_input
def roche_fluid(R_planet: u.Rjup, rho_planet: u.kg / u.m**3, rho_moon: u.kg / u.m**3):
    return 2.4423 * R_planet * np.power(rho_planet / rho_moon, 1.0 / 3.0)


@u.quantity_input
def roche_rigid(R_planet: u.Rjup, rho_planet: u.kg / u.m**3, rho_moon: u.kg / u.m**3):
    return R_planet * np.power(2.0 * rho_planet / rho_moon, 1.0 / 3.0)

def make_poly(xlim, curve):
    "return a shape defined by the y points and an x value, making a closed curve suitable for filling an area on a plot"
    fillcurve = np.append(curve, [curve[-1], curve[0]], axis=0)
    fillcurve[-1, 0] = xlim
    fillcurve[-2, 0] = xlim
    return fillcurve

mjup = 1898.0e24 * u.kg
mear = 5.97e24 * u.kg
msat = 568.4e24 * u.kg
mnep = 102.43e24 * u.kg

rv_limits = ascii.read(
    paths.data / "fifty_percent_completeness_contour.csv", data_start=1
)  # from Rico

# CHANGE THIS!
rv_future_limits = ascii.read(
    paths.data / "fifty_percent_completeness_contour_250m.s.csv", data_start=1
)  # from Rico

astrom_limits = ascii.read(
    paths.data / "confidence_table_updated.csv", data_start=1
)  # from Isabella Mascias
# message://%3cCAPfywv6JJ8Hrx+XbOpaa+wDY7LJOLQxeALUx6BxcFXCj_aGz0A@mail.gmail.com%3e

astrom_a = Ptoa(
    astrom_limits["Orbital Period (days)"] * u.day,
    Mbpb,
    astrom_limits["50% (MJup)"] * u.Mjup,
).to(u.Rjup)

astrom_massratio = (astrom_limits["50% (MJup)"] * u.Mjup / Mbpb).decompose()

x = np.logspace(0, 2.7) * u.Rjup
y = np.logspace(-5, 0)  # moon/planet mass ratio

mmoon = y * Mbpb

# given the mass, calculate the radius
rmoon = chen_mass_radius(mmoon)
V_moon = (4.0 / 3) * np.pi * np.power(rmoon, 3.0)
V_bpb = (4.0 / 3) * np.pi * np.power(Rbpb, 3.0)

rho_moon = (mmoon / V_moon).to(u.g / u.cm**3)
rho_bpb = (Mbpb / V_bpb).to(u.g / u.cm**3)

roche_r = roche_rigid(Rbpb, rho_bpb, rho_moon).to(u.Rjup)
roche_f = roche_fluid(Rbpb, rho_bpb, rho_moon).to(u.Rjup)

# calculate rhill

rh = rhill(Mstar, Mbpb, abpb)

# make the plot
# make the plot
# make the plot

import matplotlib as mpl

mpl.rcParams["axes.linewidth"] = 2


fig, ax = plt.subplots(1, 1, figsize=(11, 5))
fig.set_zorder(50)

ax.set_xscale("log")
ax.set_yscale("log")

sma_jup_lower = 1
sma_jup_upper = 3000

mfrac_lower =5e-5
mfrac_upper = 1e0

ax.set_xlim(sma_jup_lower, sma_jup_upper)
ax.set_ylim(mfrac_lower, mfrac_upper)
ax.tick_params(axis="both", which="major", labelsize=14)

ax.set_axisbelow(True)

ax.grid()
ax.grid(which="minor", color="0.9")

# RV limit

ax.plot(
    rv_limits["Semi-major axis [R$_{jup}$]"],
    rv_limits["$M_{moon}/M_p$ sin i"],
    linewidth=2,
    label="RV limit",
    color="blue",
)

vhei = 1.5

ax.fill_between(rv_limits["Semi-major axis [R$_{jup}$]"],
    rv_limits["$M_{moon}/M_p$ sin i"], y2=rv_limits["$M_{moon}/M_p$ sin i"]*vhei,
    alpha=0.2)

ax.text(70,0.043,'CRIRES+ limit', rotation = 11,
    zorder=50, color='blue',
    horizontalalignment='center', verticalalignment='bottom',
    fontsize=10)

# future RV limit

ax.plot(
    rv_limits["Semi-major axis [R$_{jup}$]"],
    rv_limits["$M_{moon}/M_p$ sin i"]/20,
    linewidth=2,
    linestyle=(0,(1,1)),
    label="Future RV limit",
    color="darkviolet",
)

ax.text(16,1e-3,'25 additional epochs with CRIRES+', rotation = 10,
    zorder=50, color='darkviolet',
    horizontalalignment='center', verticalalignment='bottom',
    fontsize=10)

# astrometric limit

ax.plot(astrom_a, astrom_massratio, 
    linewidth=2,
    color='#008080',
    label="astrometric limit")

ax.fill_between(astrom_a.value,
    astrom_massratio, y2=astrom_massratio*vhei,
    alpha=0.2, color='#008080')


ax.text(120,0.0066,'Astrometric limit', rotation = -21,
    zorder=50, color='#006060',
    horizontalalignment='center', verticalalignment='bottom',
    fontsize=10)

# calculate Roche regions

solid_poly = make_poly(-1, np.stack((roche_r.value, y), axis=1))
liquid_poly = make_poly(-1, np.stack((roche_f.value, y), axis=1))

p1 = Polygon(
    solid_poly, closed=True, facecolor="0.5", edgecolor=None, zorder=10
)

p2 = Polygon(
    liquid_poly,
    closed=True,
    facecolor="0.5",
    edgecolor=None,
    alpha=0.5,
    zorder=5,
)
ax.add_artist(p1)
ax.add_artist(p2)

# Roche limit text labels

ax.text(1.60, 0.02, "Roche limit", zorder=30, color='white',
    horizontalalignment='center', verticalalignment='center',
    rotation=-90, fontsize=24)

rochedict = dict(rotation = -52,zorder=30, color='white',
    horizontalalignment='center', verticalalignment='top',
    fontsize=12)

text_ypos=0.76

ax.text(1.5, text_ypos, "rigid", **rochedict)
ax.text(2.9, text_ypos, "fluid", **rochedict)

ax.set_ylabel("$M_{\\mathrm{moon}}/M_{\\mathrm{\\beta\\,Pic\\,b}} \\sin i$", fontsize=16)
ax.set_xlabel("Semi-major axis [$R_{\\mathrm{Jup}}$]", fontsize=16)

# planet masses

planetlines = dict(color="black", alpha=0.5, linestyle=(0, (5, 1)) )

planetlab = dict(fontsize=10, 
    verticalalignment='center', horizontalalignment='right')
bboxdict = dict(facecolor='white', 
    alpha=0.9, linewidth=0, edgecolor=None)

# label the planet masses

(low, upp) = ax.get_xlim()

toff = 200

ax.hlines((mear / Mbpb).decompose(), low, upp, **planetlines)
ax.text(upp-toff, (mear / Mbpb).decompose(), "Earth", bbox=bboxdict, **planetlab)

ax.hlines((msat / Mbpb).decompose(), low, upp, **planetlines)
ax.text(upp-toff, (msat / Mbpb).decompose(), "Saturn", bbox=bboxdict, **planetlab)

ax.hlines((mnep / Mbpb).decompose(), low, upp, **planetlines)
ax.text(upp-toff, (mnep / Mbpb).decompose(), "Neptune", bbox=bboxdict, **planetlab)

ax.hlines((1 * u.Mjup / Mbpb).decompose(), low, upp, **planetlines)
ax.text(upp-toff, (1 * u.Mjup / Mbpb).decompose(), "Jupiter", bbox=bboxdict, **planetlab)

# Hill sphere stability limits


# % aE ≈ 0.4895 (1.0000− 1.0305eP− 0.2738esat) Domingos 2006 Eq. 5 
# % take e=0.10 gives 0.44 Rhill
stable = 0.44
ax.vlines((rh * stable).to(u.Rjup).value, -5, 2, color="darkred",linewidth=3)

ax.add_artist(
    Rectangle(((rh * stable).to(u.Rjup).value,-5), 400, 7,
    alpha=0.5, facecolor='darkred', edgecolor=None, zorder=40 ))

# retrograde stable orbits in Hill sphere
#ax.vlines((rh * 0.6).to(u.Rjup).value, -5, 2, color="black")

ax.text((rh * stable).to(u.Rjup).value+80, 6.5e-3, 
    "Stable prograde orbits in Hill sphere", rotation = 90,
    zorder=50, color='white',
    horizontalalignment='left', verticalalignment='center',
    fontsize=14)



# Poon 2024 limits
# Poon 2024 limits
# Poon 2024 limits

poon_lower = 0.03 * u.au 
poon_upper = 0.05 * u.au 

poon_mass_lower = 1.0 * mnep 

# ax.add_artist(
#     Rectangle(((poon_lower).to(u.Rjup).value, (poon_mass_lower/Mbpb).decompose() ), (poon_upper-poon_lower).to(u.Rjup).value, 0.005,
#     alpha=0.5, facecolor='orange', edgecolor=None, zorder=40 ))
# ax.add_artist(
#     Rectangle(((poon_lower).to(u.Rjup).value, (poon_mass_lower/Mbpb).decompose() ), (poon_upper-poon_lower).to(u.Rjup).value, 0.002,
#     alpha=0.5, facecolor='orange', edgecolor=None, zorder=40 ))

ax.add_artist(
    Rectangle(((poon_lower).to(u.Rjup).value, (poon_mass_lower/Mbpb).decompose() ),
    (poon_upper-poon_lower).to(u.Rjup).value, 0.0005,
    facecolor='brown', edgecolor=None, zorder=40 ))

poon_mass_mid = (0.039 * u.au)

ax.annotate("", xytext=(poon_mass_mid.to(u.Rjup).value,(poon_mass_lower/Mbpb).decompose()),
    xy=(poon_mass_mid.to(u.Rjup).value, (poon_mass_lower/Mbpb).decompose()+0.005),
            arrowprops=dict(facecolor="brown",edgecolor=None,linewidth=0),
            )

ax.text(72,7e-3, "Poon+ (2024)",
    horizontalalignment='right', color="brown",
     verticalalignment='center' )


# put orbital period on the upper axis

axp = ax.twiny()
axp.set_xlabel("Period [d]",fontsize=16)

axp.set_xscale("log")

period_lower = atoP(sma_jup_lower * u.Rjup, Mbpb, 1e-1 * Mbpb).to(u.d)
period_upper = atoP(sma_jup_upper * u.Rjup, Mbpb, 1e-1 * Mbpb).to(u.d)

axp.set_xlim(period_lower.value, period_upper.value)

# Kipping 2022 NatAst - Kepler 1708 b-i

k1708_P = 4.6 * u.d
k1708_massratio = 0.11

# axp.scatter(k1708_P, k1708_massratio, color='black', marker="s")


# axp.text(k1708_P.value-0.5, k1708_massratio, "1708 b-i",
#     horizontalalignment='right', verticalalignment='center' )


# Teachey 2018 SciAdv - Kepler 1625 b-i

# k1625_P = 22 * u.d # +17 -9
# k1625_massratio = 0.0141 # +0.0048 -0.0039

# axp.scatter(k1625_P, k1625_massratio, color='black', marker="o")

# axp.text(k1625_P.value-2.5, k1625_massratio, "1625 b-i",
#     horizontalalignment='right', verticalalignment='center' )


# put Earth masses on right hand axis

axme = ax.twinx()
axme.set_ylabel("$M_{\\mathrm{moon}}/M_\\oplus \\sin i$",fontsize=16)

axme.set_yscale("log")

mearth_lower = (mfrac_lower * Mbpb).to(u.Mearth)
mearth_upper = (mfrac_upper * Mbpb).to(u.Mearth)

axme.set_ylim(mearth_lower.value, mearth_upper.value)

plt.draw()

plt.savefig(paths.figures / "rbpic_exomoon_limits.pdf", bbox_inches='tight')

if debug:
    plt.show()
