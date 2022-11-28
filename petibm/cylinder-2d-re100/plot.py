"""Plot the history of the force coefficients."""

import pathlib
import numpy
import scipy.interpolate
from h5py import File as h5open
from matplotlib import pyplot
from scipy import fftpack, signal

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[2].joinpath("resources", "figstyle"))

# Load forces from file.
rootdir = pathlib.Path(__file__).resolve().parents[1]
casedir = pathlib.Path(__file__).resolve().parents[0]

# force data
data = numpy.loadtxt(casedir.joinpath("output", "forces-0.txt"))

# convert force to force coefficients
data[:, 1:] = data[:, 1:] * 2.

# Define time interval used to compute stats.
time_limits = (150.0, 200.0)
print('Initial time interval: {}'.format(time_limits))
mask = numpy.where((data[:, 0] >= time_limits[0]) & (data[:, 0] <= time_limits[1]))[0]

# Compute the minima and maxima of the lift coefficient.
idx_min = signal.argrelextrema(data[:, 2], numpy.less_equal, order=5)[0]
idx_min = numpy.intersect1d(idx_min, mask, assume_unique=True)
print('min(CL) = {}'.format(data[idx_min, 2]))
idx_max = signal.argrelextrema(data[:, 2], numpy.greater_equal, order=5)[0]
idx_max = numpy.intersect1d(idx_max, mask, assume_unique=True)
print('max(CL) = {}'.format(data[idx_max, 2]))

# Redefine time interval between first minima and last maxima.
time_limits = (data[idx_min[0], 0], data[idx_max[-1], 0])
print('Time interval: {}'.format(time_limits))

# Compute the time-averaged force coefficients.
cd_mean = numpy.mean(data[idx_min[0]:idx_max[-1]+1, 1])
cl_mean = numpy.mean(data[idx_min[0]:idx_max[-1]+1, 2])
print('<CD> = {:.4f}'.format(cd_mean))
print('<CL> = {:.4f}'.format(cl_mean))

# Compute the RMS of the lift coefficient.
cl2 = data[:, 2][idx_min[0]:idx_max[-1] + 1]
rms = numpy.sqrt(numpy.mean(numpy.square(cl2)))
print('rms(CL) = {:.4f}'.format(rms))

# Compute the Strouhal number using lift coefficient.
dt = data[1, 0] - data[0, 0]
fft = scipy.fft.fft(cl2)
freqs = fftpack.fftfreq(len(cl2), dt)
idx = numpy.argmax(abs(fft))
strouhal = freqs[idx]
print('St = {:.4}'.format(strouhal))

# use the mean pressure as the reference pressure
with h5open(casedir.joinpath("output", "0020000.h5"), "r") as h5file:
    pref = h5file["p"][...].mean()

# get probe data
with h5open(casedir.joinpath("output", "probe-p.h5"), "r") as h5file:
    x = h5file["mesh/x"][...]
    y = h5file["mesh/y"][...]
    ids = numpy.argsort(h5file["mesh/IS"][...].flatten())
    p = h5file[f"p/{max(h5file['p'].keys(), key=float)}"][...].flatten()[ids].reshape(y.size, x.size)
    # p -= pref   # correct the pressure

# create an interpolater for surface pressure from PetIBM
probe_interp = scipy.interpolate.RectBivariateSpline(x, y, p.T)

# note for PetIBM's diffused immersed boundary, we use r+3dx as the cylinder surface
degrees = numpy.linspace(0., numpy.pi, 361)
surfp: numpy.zeros(361)
surfx = (0.5 + 4 * (x[1] - x[0])) * numpy.cos(degrees)
surfy = (0.5 + 4 * (x[1] - x[0])) * numpy.sin(degrees)

# interpolation
surfp = probe_interp(surfx, surfy, grid=False) * 2  # Cp = (p-p_ref) * 2

# back suction pressure coefficient
print("Back suction coefficient:", surfp[0])

# convert from radius to degrees
degrees = degrees * 180 / numpy.pi

# li et al, 2016
li_surfp = numpy.loadtxt(casedir.joinpath("li_et_al_2016_cylinder2dRe100_cp.csv"), delimiter=",")
li_surfp[:, 0] = 180. - li_surfp[:, 0]

# plot
fig = pyplot.figure(figsize=(6.5, 3))
fig.suptitle(r"PetIBM, 2D cylinder, $Re=100$")
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.5])

ax = fig.add_subplot(gs[0, 0])
ax.set_title("Lift and drag coefficient")
ax.set_xlabel(r'Non-dimensional time, $T$')
ax.set_ylabel('Force coefficients')
ax.grid()
ax.plot(data[:, 0], data[:, 1], label=r'$C_D$')
ax.plot(data[:, 0], data[:, 2], label=r'$C_L$')
ax.legend(ncol=2)
ax.set_xlim(data[0, 0], data[-1, 0])
ax.set_ylim(-0.5, 1.5)

ax = fig.add_subplot(gs[0, 1])
ax.set_title(r"Surface $C_p$ at $T=200$")
ax.set_xlabel(r"Degree (ccw. from $+x$)")
ax.set_ylabel(r"Pressure coefficient, $C_p$")
ax.plot(degrees, surfp, label="PetIBM", lw=2)
ax.plot(li_surfp[:, 0], li_surfp[:, 1], label="Li et al., 2016", alpha=0.8, ls="-.", lw=2)
ax.legend(loc=0)
ax.grid()

fig.savefig(rootdir.joinpath("figures", "petibm-cylinder-2d-re100-val.png"))
