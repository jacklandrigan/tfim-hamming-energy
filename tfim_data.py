import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tfim
from tfim_hd_energy import ground_states


seeds = open("seeds.txt", "r")
for line in seeds:
    item = line.strip()
    print(item)
    ind = item.index("_")
    N = int(item[1:ind])
    S = int(item[ind+2:])
    output = "Data/" + item
    #subprocess.call(["/opt/anaconda2/bin/python", "tfim_diag.py", str(N), "--save_state", "--h_min", "-4", "--h_max", "2", "--h_log", "True", "--h_log_num", "25", "-S", str(S), "--model", "SK", "-o", output])
    
    lattice = tfim.Lattice([N])
    basis = tfim.IsingBasis(lattice)
    Jij = tfim.Jij_instance(N, 1, dist="bimodal", seed=S)
    energy = tfim.JZZ_SK(basis, Jij)
    grounds = ground_states(N, energy)
    ind1 = grounds[0] + 1
    ind2 = grounds[1] + 1
    psi0 = (np.loadtxt(output + "_psi0.dat"))
    diff = np.abs((psi0[:,ind1])**2-(psi0[:,ind2])**2)
    
    plt.plot(psi0[:,0], diff, color="b")
    plt.xscale("log")
    plt.savefig("Graphs/" + item + "_1")
    plt.clf()
    
    plt.plot(psi0[:,0], np.abs(np.max(diff)-diff), color="r")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("Graphs/" + item + "_2")
    plt.clf()


    




