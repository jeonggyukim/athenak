"""
Radiative supernova remnant with the NCR chemical network: 1e51 erg and 10 Msun in a
uniform medium of n_H = 1 cm^-3, 64^3 cells over 64 pc, run to t = 0.05 (pc/(km/s)).
Shell formation is the maximum of the hot-gas mass M_hot (T > 2e4 K); its time t_sf, the
hot mass M_sf and the radial momentum p_sf are compared with this code's own reference
run (inputs/chemistry2/radiative_snr_ncr.athinput, Mac CPU, g++-16; identical on 1 and 4
MPI ranks).

The same run against Tigris NCR (tst/regression feedback/rad_snr, 4 ranks): t_sf -0.01%,
M_sf +1.4%, p_sf +0.1%.
"""

import numpy as np

import test_suite.testutils as testutils

input_file = "../../../inputs/chemistry2/radiative_snr_ncr.athinput"

# code units of the input: length 1 pc, time 1 pc/(km/s), mass mu_H m_H cm^-3 pc^3
U_M = 6.882615081124725e31/1.98841586e33   # Msun
U_V = 3.0856775809623245e18/3.08567758e13/1e5   # km/s

reference = {"t_sf": 0.0384684, "M_sf": 1503.91, "p_sf": 1.93076e5}
tolerance = {"t_sf": 0.005, "M_sf": 0.01, "p_sf": 0.005}


def read_hst(path):
    with open(path) as f:
        f.readline()
        names = [s.split("=")[1] for s in f.readline().split()[1:]]
    data = np.loadtxt(path)
    return {n: data[:, i] for i, n in enumerate(names)}


def test_radiative_snr_cpu():
    try:
        assert testutils.run(input_file, ["time/tlim=0.05", "output2/dt=1.0"])
        h = read_hst("snr.user.hst")
        i = int(np.argmax(h["M_hot"]))
        result = {"t_sf": h["time"][i], "M_sf": h["M_hot"][i]*U_M,
                  "p_sf": h["pr"][i]*U_M*U_V}
        for k, ref in reference.items():
            err = abs(result[k]/ref - 1.0)
            assert err < tolerance[k], (
                f"{k} = {result[k]:.6g}, reference {ref:.6g}, relative error "
                f"{err:.2e} > {tolerance[k]}")
    finally:
        testutils.cleanup()
        testutils.run_command(["sh", "-c", "rm -rf snr.*.hst bin"])
