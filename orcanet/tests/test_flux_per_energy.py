import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

arr_energy_correct = np.load('/home/woody/capn/mppi033h/Code/HPC/cnns/results/plots/saved_predictions/arr_energy_correct_model_VGG_4d_xyz-t_and_yzt-x_and_4d_xyzt_muon-CC_to_elec-CC_multi_input_single_train_tight-1_tight-2_lr0.003_dense128-32.npy')

bins = np.arange(3, 41, 1)
bin_contents_numu = np.zeros(len(bins) - 1, dtype=np.uint64)
bin_contents_nue = np.zeros(len(bins) - 1, dtype=np.uint64)

for i in range(len(arr_energy_correct)):
    if i % 10000 == 0: print(i)
    energy = arr_energy_correct[i, 0]
    particle_type = arr_energy_correct[i, 2]
    is_cc = arr_energy_correct[i, 3]

    for j in range(len(bin_contents_numu)):
        if bins[j] < energy <= bins[j+1]:

            if particle_type == 14 and is_cc == 1:
                bin_contents_numu[j] += 1
                break

            if particle_type == 12 and is_cc == 1:
                bin_contents_nue[j] += 1
                break

print(bin_contents_numu)
print(bin_contents_nue)
bin_contents_stat_acc = np.zeros(len(bins) - 1, dtype=np.uint64)
for i in range(len(bin_contents_numu)):
    bin_contents_stat_acc[i] =  bin_contents_numu[i] / float(bin_contents_numu[i] + bin_contents_nue[i])

hist_stat_acc = plt.hist(bin_contents_stat_acc, bins=len(bin_contents_numu))
plt.savefig('test.pdf')


