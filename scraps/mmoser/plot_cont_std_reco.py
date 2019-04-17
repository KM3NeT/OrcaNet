import h5py
import numpy as np
from matplotlib import pyplot as plt


def make_cut_file(fpath_col_names, meta_filepath, summary_file_savepath):
    col_names = {}

    with open(fpath_col_names) as f_col_names:
        cols_list = f_col_names.read().splitlines()

    for i, col in enumerate(cols_list):
        col_names[col] = i

    selected_cols = ('is_cc', 'is_neutrino', 'type', 'muon_score', 'dusj_is_selected', 'gandalf_is_selected',
                     'gandalf_loose_is_selected', 'oscillated_weight_one_year')

    usecols = [col_names[col_name] for col_name in selected_cols]
    usecols.sort()

    print('Loading the summary file content')
    dtype = dict()
    dtype['names'] = selected_cols
    dtype['formats'] = (np.float64, ) * len(selected_cols)
    summary_file_arr = np.loadtxt(meta_filepath, delimiter=' ', usecols=usecols, dtype=dtype)

    f = h5py.File(summary_file_savepath, 'w')
    print('Saving the summary file content')
    f.create_dataset('summary_array', data=summary_file_arr, compression='gzip', compression_opts=1)
    f.close()


def make_plots():

    # get files
    fpath = '/home/saturn/capn/mppi033h/Data/standard_reco_files/verify_contamination/summary_file_cut.h5'
    f = h5py.File(fpath, 'r')
    #w_1_y = np.load('/home/saturn/capn/mppi033h/Data/standard_reco_files/weight_1_year_col.npy')
    w_1_y = f['summary_array']['oscillated_weight_one_year']

    # apply event selection (no reco lns)
    s = f['summary_array']
    # dusj_is_selected = s['dusj_is_selected']
    # gandalf_loose_is_selected = s['gandalf_loose_is_selected']
    # selection_classifier_bg = np.logical_or(dusj_is_selected, gandalf_loose_is_selected)
    # s = s[selection_classifier_bg]
    # w_1_y = w_1_y[selection_classifier_bg]

    # get is_neutrino, is_mupage info
    ptype, is_cc = s['type'], s['is_cc']
    is_neutrino = np.logical_or.reduce((np.abs(ptype) == 12, np.abs(ptype) == 14, np.abs(ptype) == 16))
    is_mupage = np.abs(ptype) == 13

    n_neutrinos_total = np.count_nonzero(is_neutrino)
    n_mupage_total = np.count_nonzero(is_mupage)
    print('Neutrinos total: ' + str(n_neutrinos_total) + '\nMupage total: ' + str(n_mupage_total))

    # get plot data
    muon_score = s['muon_score']
    n_neutrinos_total_weighted = np.sum(w_1_y[is_neutrino])
    print('Neutrinos total weighted: ' + str(n_neutrinos_total_weighted))

    cuts = np.linspace(0, 1.01, 102)
    ax_muon_contamination, ax_neutrino_efficiency = [], []
    for i in range(len(cuts)):
        if i % 100 == 0:
            print(i)

        # get mask for events that survive the cut
        neutrino_sel = muon_score[is_neutrino] < cuts[i]
        mupage_sel = muon_score[is_mupage] < cuts[i]

        # get weighter leftover event number
        n_neutrino_weighted = np.sum(w_1_y[is_neutrino][neutrino_sel])
        n_mupage_weighted = np.sum(w_1_y[is_mupage][mupage_sel])

        print('----------------------------')
        print('Neutrinos weighted: ' + str(n_neutrino_weighted))
        print('Mupage weighted: ' + str(n_mupage_weighted))
        print('----------------------------')

        # calculate contamination and efficiency and add to lists
        if n_neutrino_weighted != 0:
            muon_contamination = (n_mupage_weighted / n_neutrino_weighted) * 100
            neutrino_efficiency = (n_neutrino_weighted / n_neutrinos_total_weighted) * 100

            ax_muon_contamination.append(muon_contamination)
            ax_neutrino_efficiency.append(neutrino_efficiency)


    # make plot
    fig, ax = plt.subplots()
    ax.plot(np.array(ax_muon_contamination), np.array(ax_neutrino_efficiency), label='Std Reco')
    ax.set_xlim(left=0, right=20)
    ax.set_xlabel('Muon contamination [%]'), ax.set_ylabel('Neutrino Efficiency [%]')
    ax.grid(True)
    ax.legend(loc='upper right')

    plt.savefig('/home/saturn/capn/mppi033h/Data/muon_contamination_std.pdf')


if __name__ == '__main__':
    fpath_col_names = '/home/saturn/capn/mppi033h/Data/standard_reco_files/verify_contamination/reduced_pid_output_shifted_04_18_withStandardWeight_column_names.txt'
    meta_filepath = '/home/saturn/capn/mppi033h/Data/standard_reco_files/verify_contamination/reduced_pid_output_shifted_04_18_withStandardWeight.meta'
    summary_file_savepath = '/home/saturn/capn/mppi033h/Data/standard_reco_files/verify_contamination/summary_file_cut.h5'

    #make_cut_file(fpath_col_names, meta_filepath, summary_file_savepath)
    make_plots()






