import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# compare energy spectra of 1-5GeV prod and 3-100GeV prod

input_files = {'muon-CC': ['/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_1-5GeV/4dTo4d/'
           'time_-250+500_geo-fix_60b/JTE_KM3Sim_gseagen_muon-CC_1-5GeV-9_2E5-1bin-1_0gspec_ORCA115_9m_2016_9_xyzt.h5',
           '/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/'
           'time_-250+500_geo-fix_60b/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_296_xyzt.h5'],
               'elec-CC': ['/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_1-5GeV/4dTo4d/'
           'time_-250+500_geo-fix_60b/JTE_KM3Sim_gseagen_elec-CC_1-5GeV-2_7E5-1bin-1_0gspec_ORCA115_9m_2016_512_xyzt.h5',
           '/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/'
           'time_-250+500_geo-fix_60b/JTE_KM3Sim_gseagen_elec-CC_3-100GeV-1_1E6-1bin-3_0gspec_ORCA115_9m_2016_852_xyzt.h5'],
               'elec-NC': ['/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_1-5GeV/4dTo4d/'
           'time_-250+500_geo-fix_60b/JTE_KM3Sim_gseagen_elec-NC_1-5GeV-2_2E6-1bin-1_0gspec_ORCA115_9m_2016_44_xyzt.h5',
           '/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/'
           'time_-250+500_geo-fix_60b/JTE_KM3Sim_gseagen_elec-NC_3-100GeV-3_4E6-1bin-3_0gspec_ORCA115_9m_2016_573_xyzt.h5']}

pdf_plots = PdfPages('energy_spectra.pdf')
fig, axes = plt.subplots()

for ptype, data_files in input_files.items():

    f_prod_1_to_5 = h5py.File(data_files[0], 'r')
    #print f_prod_1_to_5['y'].shape[0]
    f_prod_3_to_100 = h5py.File(data_files[1], 'r')
    #print f_prod_3_to_100['y'].shape[0]

    #e_1_to_5 = f_prod_1_to_5['y'][:, 2]
    ratio = 0.75
    n_events_e_1_to_5 = f_prod_1_to_5['y'].shape[0]
    last_event = int(ratio * n_events_e_1_to_5)
    e_1_to_5 = f_prod_1_to_5['y'][0:last_event, 2]

    e_3_to_100 = f_prod_3_to_100['y'][:, 2]

    # 1-5 GeV
    hist_1_to_5 = plt.hist(e_1_to_5, bins=100)
    plt.title(ptype)
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Counts [#]')
    pdf_plots.savefig(fig)

    plt.cla()

    # 3-100 GeV
    hist_3_to_100 = plt.hist(e_3_to_100, bins=100)
    plt.title(ptype)
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Counts [#]')
    pdf_plots.savefig(fig)

    plt.cla()

    # Combined

    remove_3_to_5_GeV = True

    if remove_3_to_5_GeV is True:
        e_1_to_5 = e_1_to_5[e_1_to_5 < 3]

    print(e_1_to_5.shape[0])

    e_total = np.concatenate([e_1_to_5, e_3_to_100], axis=0)
    plt.title(ptype)
    hist_total = plt.hist(e_total, bins=100)
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Counts [#]')
    plt.axvline(x=3, color='black', linestyle='--')

    #plt.xlim((0,10))
    #plt.xticks(np.arange(0, 21, 1))
    pdf_plots.savefig(fig)
    plt.cla()

pdf_plots.close()
plt.close()