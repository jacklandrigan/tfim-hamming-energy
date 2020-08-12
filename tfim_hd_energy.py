import tfim
import numpy as np
import matplotlib.pyplot as plt

#################################################################################

def ground_states(N, energy):
    """returns list of ground state configuration indices for a random instance of 
    N-spin system"""
    energy1 = energy.tocsr()
    ground = [0]
    for i in range(1,2**N):
        if energy1[i,i] > energy1[ground[0],ground[0]]:
            ground = [i]
        elif energy1[i,i] == energy1[ground[0],ground[0]]:
            ground.append(i)
    return ground

#################################################################################

def hamming_distance(basis, ind1, ind2):
    """returns hamming distance between two configurations of an N-spin system"""
    state1 = basis.state(ind1)
    state2 = basis.state(ind2)
    nonzero = np.nonzero(state1-state2)
    ham_dist = len(nonzero[0])
    return ham_dist

#################################################################################

def all_ham_dist(N, basis, ind):
    """returns list of hamming distance values between configuration given by index 
    and all other configurations in N-spin system"""
    hd_list = []
    for i in range(2**N):
        hd_list.append(hamming_distance(basis, ind, i))
    return hd_list

#################################################################################

def energy_states(N, energy):
    """returns list of which energy state each configuration of N-spin system is in"""
    energy1 = energy.tocsr()
    values = []
    states = []
    for i in range(2**N):
        if energy1[i,i] not in values:
            values.append(energy1[i,i])
            values.sort(reverse=True)
    for j in range(2**N):
        index = values.index(energy1[j,j])
        states.append(index)
    return states

#################################################################################

def hamming_energy_array(N, basis, energy, ind):
    """returns array with count of configurations at given energy level and hamming distance 
    from configuration given by ind (ex: if there are 6 configs in energy level 3 at hamming 
    distance 4, array[3,4] = 6)"""
    states = energy_states(N, energy)
    hd_list = all_ham_dist(N, basis, ind)
    dim1 = max(states) +1
    dim2 = max(hd_list) +1
    arr = np.empty([dim1,dim2])
    arr.fill(0)
    for i in range(len(hd_list)):
        ind1 = states[i]
        ind2 = hd_list[i]
        arr[ind1,ind2] +=1
    return(arr)

#################################################################################

def ground_arrays(N, basis, energy):
    """returns list containing a hamming distance/energy level array for each ground
    state configuration"""
    ground_list = ground_states(N, energy)
    array_list = []
    for i in range(len(ground_list)):
        arr = hamming_energy_array(N, basis, energy, ground_list[i])
        array_list.append(arr)
    return array_list

#################################################################################

def histogram(array, title, xticks, yticks):
    hd1 = array[:,1]
    hd2 = array[:,2]
    max1 = np.max(np.nonzero(hd1)) +1
    max2 = np.max(np.nonzero(hd2)) +1
    plt.hist(range(0,max1), max1, weights=hd1[0:max1], label="hd1", align="left", range=(0,max1), color='g')
    plt.hist(range(0,max2), max2, weights=hd2[0:max2], label="hd2", align="left", range=(0,max2), color='b')
    plt.xlabel("Energy Level")
    plt.ylabel("Count")
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.legend(loc="upper right")
    plt.title(title)
    path = "/Users/jacklandrigan/Desktop/tfim-hamming-energy-master/Histograms/" + title + ".png"
    plt.savefig(path)
    plt.show()