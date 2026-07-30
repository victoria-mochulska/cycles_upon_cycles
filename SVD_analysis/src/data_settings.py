import tifffile
import pandas as pd
import numpy as np

data_dir = '../raw_data_kymos/'

sample_names = {
    'TE_22C_14ss': {'20210301': [1, 2, 4, 5, 6, 7]},
    'TE_22C_15ss': {'20200817': [1, 4, 6, 7, 8, 9]},
    'TE_27C_14ss': {'20200815': [1, 2, 3],
                    '20210310': [2, 3, 4, 5, 6, 7]},
    # 'TE_27C_14ss_wt': {'20210610': [2, 4, 5, 6, 7, 9, 10]},
    'TE_27C_17ss': {'20200815': [6, 8, 9, 10]},
    # 'TE_32C_15ss': {'20210224': [1, 2, 3, 4, 6, 9, 10],
    #                 '20210325': [1, 3, 4, 5, 6, 7, 8]},
    'TE_32C_14ss_5dt': {'20220428': [1, 2, 3, 4, 5],
                        '20220429': [1, 2, 3, 4, 5],
                        '20220430': [1, 2, 3, 4, 5]}, }

sample_names_catenin = {
    'ctnnb1-venus_21-27C_15ss': {'20210518': [1, 2, 3, 4],
                                 '20210522': [1, 2, 3, 4]},
    'ctnnb1-venus_22C_15ss': {'20210520': [1, 2, 3, 4],
                              '20210521': [1, 3, 4]},
    'ctnnb1-venus_27C_15ss': { #'20210209': [1, 2],
                              '20210516': [1, 2, 3]},
    'ctnnb1-venus_32C_15ss': {'20210514': [1, 2, 4],
                              '20210515': [1, 2, 3, 4]}
}

sample_cropping_catenin = {
    'ctnnb1-venus_21-27C_15ss': {'20210518': [1, 1, 5, 1],
                                 '20210522': [1, 1, 1, 12]},
    'ctnnb1-venus_22C_15ss': {'20210520': [1, 1, 1, 1],
                              '20210521': [25, 30, 25]},
    'ctnnb1-venus_27C_15ss': { #'20210209': [1, 2],
                              '20210516': [20, 25, 18]},
    'ctnnb1-venus_32C_15ss': {'20210514': [55, 55, 50],
                              '20210515': [1, 25, 50, 50]}
}


sample_names_het = {'TE_27C_14ss_het': {'20210610': [1, 3, 8]}, }


sample_names_cycling = {'TE_21-27C12h_15ss': {'20210604': [1, 2, 3],
                                              '20210605': [1, 2, 3, 4, 5, 6, 7, 8]},
                        'TE_21-27C_13ss': {'20210404': [1, 2, 3, 4]},
                        'TE_21-27C_14ss': {'20210408': [1, 2, 4, 5, 6, 7],
                                           '20210527': [1, 2, 3, 5, 7, 8]},
                        #                        'TE_21-27C_14ss_wt': {'20210530': [1, 3, 4, 5, 6],
                        #                          '20210601': [1, 2, 5, 7]},
                        #                        'TE_21-27C_14ss_het': {'20210530': [2,],
                        #                           '20210601': [3, 4, 6]},
                        }


sample_names_6h = {'221128_her7Het': {'221128': [1, 2, 3, 5, 6], },
                   '221129_her7Het': {'221129': [1, 2, 3, 4, 5, 6]},
                   '230501_5minDelT': {'230501': [1, 2, 3]},
                   }

sample_names_6h_newreg = {'221128': {'221128': [1, 2, 3, 5, 6],},
                          '221129': {'221129': [1, 2, 3, 4, 5]}
                          }

sample_names_newreg = {'new_registration_22C': {'20200817': [2, 3],
                                                '20210301': [1, 5, 6, 7]},
                       'new_registration_27C': {'230704': [2,],
                                                '230722': [1,],
                                                '230728': [1, 3],
                                                '230820': [2,]},
                       'new_reg_27C_betaCatenin': {'20210209': [1, 2,],
                                                   '20210516': [1, 2,]}
                       }

sample_names_long_exp = {'data_longExperiments/27C': {'230704': [2, 5, 6],
                                 '230722': [3,],
                                 '230728': [1, 3],
                                 '230820': [1, 2, 3]},
}

sample_names_hilbert = {'hilbert/6h': {'221128': [1, 2, 3, 5, 6], '221129': [1, 2, 3, 4, 5]},
                        'hilbert/12h': {'20210604': [1, 2, 3], '20210605': [1, 2, 3, 4, 5, 6]},
                        'hilbert/24h': {'20210408': [5, 7], '20210527': [4, 5, 8]}
}

# Only 6h hilbert samples
sample_names_hilbert_6h = {'hilbert/6h': {'221128': [1, 2, 3, 6], '221129': [1, 3, 4, 5]}}


sample_temperatures = {
    'TE_22C_14ss': 22.,
    'TE_22C_15ss': 22.,
    'TE_27C_14ss': 27.,
    'TE_27C_17ss': 27.,
    'TE_32C_15ss': 32.,
    'TE_32C_14ss_5dt': 32.,

    'TE_27C_14ss_het': 27.,
    'TE_27C_14ss_wt': 27.,

    'ctnnb1-venus_21-27C_15ss': 24.,
    'ctnnb1-venus_22C_15ss': 22.,
    'ctnnb1-venus_27C_15ss': 27.,
    'ctnnb1-venus_32C_15ss': 32.,
    'new_registration_22C': 22.,
    'new_registration_27C': 27.,
    'new_reg_27C_betaCatenin': 27.,
    'data_longExperiments/27C': 27.,

    'hilbert/6h': 32.,
    'hilbert/12h': 27.,
    'hilbert/24h': 22.,
}


sample_cropping = {
    'TE_22C_14ss': 10,
    'TE_22C_15ss': 10,

    'TE_27C_14ss': 20,
    'TE_27C_17ss': 20,

    'TE_32C_15ss': 35,
    'TE_32C_14ss_5dt': 30,

    'TE_27C_14ss_het': 10,
    'TE_27C_14ss_wt': 10,
    'data_longExperiments/27C': 10,
    'hilbert/6h': 0,
    'hilbert/12h': 0,
    'hilbert/24h': 0,
}


sample_cropping_detailed = {
    'TE_22C_14ss': {'20210301': [10, 10, 10, 10, 10, 10]},
    'TE_22C_15ss': {'20200817': [10, 10, 10, 10, 10, 10]},

    'TE_27C_14ss': {'20200815': [45, 20, 40],
                    '20210310': [50, 15, 15, 20, 30, 15]},

    'TE_27C_17ss': {'20200815': [12, 25, 25, 15]},

    'TE_32C_14ss_5dt': {'20220428': [40, 25, 40, 40, 40],
                        '20220429': [30, 15, 35, 25, 35],
                        '20220430': [40, 10, 25, 44, 16]}, }


sample_dt = {
    'TE_22C_14ss': 10.,
    'TE_22C_15ss': 10.,
    'TE_27C_14ss': 10.,
    'TE_27C_17ss': 10.,
    'TE_32C_15ss': 10.,
    'TE_32C_14ss_5dt': 5.,

    'TE_27C_14ss_het': 10.,
    'TE_27C_14ss_wt': 10.,

    'TE_21-27C12h_15ss': 10.,
    'TE_21-27C_13ss': 10.,
    'TE_21-27C_14ss': 10.,
    'TE_21-27C_14ss_wt': 10.,

    '221128_her7Het': 10.,
    '221129_her7Het': 10.,
    '230501_5minDelT': 5.,
    '221128': 10.,
    '221129': 10.,
    'new_registration_22C': 10.,
    'new_registration_27C': 10.,
    'new_reg_27C_betaCatenin': 10.,
    'data_longExperiments/27C': 10.,
    'hilbert/6h': 10.,
    'hilbert/12h': 10.,
    'hilbert/24h': 10.,
}


sample_dx = {
    'TE_22C_14ss': 1.38,
    'TE_22C_15ss': 1.38,
    'TE_27C_14ss': 1.38,
    'TE_27C_17ss': 1.38,
    'TE_32C_15ss': 1.38,
    'TE_32C_14ss_5dt': 1.38,

    'TE_27C_14ss_het': 1.38,
    'TE_27C_14ss_wt': 1.38,

    'TE_21-27C12h_15ss': 1.38,
    'TE_21-27C_13ss': 1.38,
    'TE_21-27C_14ss': 1.38,
    'TE_21-27C_14ss_wt': 1.38,

    '221128_her7Het': 1.38,
    '221129_her7Het': 1.38,
    '230501_5minDelT': 1.38,
    '221128': 1.38,
    '221129': 1.38,
    'new_registration_22C': 1.,
    'new_registration_27C': 1.,
    'new_reg_27C_betaCatenin': 1.,

    'data_longExperiments/27C': 1.38,

    'hilbert/6h': 1.38*2,
    'hilbert/12h': 1.38,
    'hilbert/24h': 1.38,
}


sample_somite_stage = {
    'TE_22C_14ss': 14,
    'TE_22C_15ss': 15,
    'TE_27C_14ss': 14,
    'TE_27C_17ss': 17,
    'TE_32C_15ss': 15,
    'TE_32C_14ss_5dt': 14,
    'TE_27C_14ss_het': 14,
    'TE_27C_14ss_wt': 14,

    'TE_21-27C12h_15ss': 15,
    'TE_21-27C_13ss': 13,
    'TE_21-27C_14ss': 14,
    'TE_21-27C_14ss_wt': 14,

    '221128_her7Het': 16,  # 16-17
    '221129_her7Het': 16,  # 16-17
    '230501_5minDelT': 16,
    '221128': 16,  # 16-17
    '221129': 16,
    'new_registration_22C': 14,  # 14-15
    'new_registration_27C': 16,  # 16-17
    'new_reg_27C_betaCatenin': 15,
    'data_longExperiments/27C': 16,
    'hilbert/6h': 16,
    'hilbert/12h': 15,
    'hilbert/24h': 14,
}
#31688e
#35b779
#ffc726

sample_colors = {
    'TE_22C_14ss': '#31688e',     #'steelblue',
    'TE_22C_15ss': '#31688e',     #tab:blue',

    'TE_27C_14ss': '#35b779',     #'forestgreen',
    'TE_27C_17ss': '#35b779',          #'lightgreen',

    'TE_32C_15ss': '#ffc726',     # 'tab:orange',
    'TE_32C_14ss_5dt': '#ffc726',  #'gold',

    'TE_27C_14ss_het': 'cyan',
    'TE_27C_14ss_wt': 'darkturquoise',

    'TE_21-27C12h_15ss': 'indianred',
    'TE_21-27C_13ss': 'tab:blue',
    'TE_21-27C_14ss': 'steelblue',
    'TE_21-27C_14ss_wt': 'silver',

    '221128_her7Het': 'forestgreen',  #
    '221129_her7Het': 'steelblue',  #
    '230501_5minDelT': 'lightgreen',
    '221128': 'tab:green',  #
    '221129': 'forestgreen',
    'new_registration_22C': 'steelblue',
    'new_registration_27C': 'forestgreen',
    'new_reg_27C_betaCatenin': 'lightgreen',
    'data_longExperiments/27C': 'darkseagreen',
    'hilbert/6h': 'forestgreen',
    'hilbert/12h': 'indianred',
    'hilbert/24h': 'steelblue',
}


def load_kymo(kymo_type, data_dir, exp, date, sample, naming = 'default'):
    path = None
    if naming == 'default':
        if kymo_type == 'mask':
            folder = 'binarykymo'
            file = 'Binary_Kymo_'
            dtype = np.uint8
        if kymo_type == 'intensity':
            folder = 'intensitykymo'
            file = 'Kymo_'
            dtype = np.uint16
        if kymo_type == 'detrended':
            folder = 'detrendedkymo'
            file = 'detrended_Kymo_'
            dtype = np.uint32
        if kymo_type == 'TL':
            folder = 'TLkymo'
            file = 'TL_Kymo_'
            dtype = np.uint16
        if kymo_type == 'phase':
            folder = 'phasekymo'
            file = 'phase_Kymo_'
            dtype = np.uint32
        if kymo_type == 'period':
            folder = 'periodkymo'
            file = 'period_Kymo_'
            dtype = np.uint32
        if kymo_type == 'catenin':
            folder = ''
            file = 'Kymo_'
            dtype = np.uint32
    if naming == 2:
        if kymo_type == 'mask':
            folder = 'binaryMask_manual'
            path = data_dir + exp + '/' + folder + '/' + 'P' + str(sample)+'.tif'
            dtype = np.uint8
            print(path)
            return ~ tifffile.imread(path, dtype=dtype)
        if kymo_type == 'intensity':
            folder = 'intensityKymo'
            file = 'intensityKymo_'
            dtype = np.uint16
        if kymo_type == 'detrended':
            folder = 'detrendedKymo'
            file = 'detrended_intensityKymo_'
            dtype = np.uint32
        if kymo_type == 'TL':
            folder = 'tlKymo'
            file = 'tlKymo_'
            dtype = np.uint16
        if kymo_type == 'phase':
            folder = 'phaseKymo'
            file = 'phase_intensityKymo_'
            dtype = np.uint32
        if kymo_type == 'period':
            folder = 'periodKymo'
            file = 'period_intensityKymo_'
            dtype = np.uint32
    if naming == 'prereg':     # pre-registered kymos
        if kymo_type == 'intensity' or kymo_type == 'detrended': # no detrended kymos
            folder1 = 'registered_intensity_kymos'
            folder2 = 'post_intensity'
            file = 'post_intensity_Kymo_'
            dtype = np.uint16
        if kymo_type == 'phase':
            folder1 = 'registered_intensity_kymos'
            folder2 = 'phasekymo'
            file = 'phase_post_intensity_Kymo_'
            dtype = np.uint16
        if kymo_type == 'period':
            folder1 = 'registered_intensity_kymos'
            folder2 = 'periodkymo'
            file = 'period_post_intensity_Kymo_'
            dtype = np.uint16
    if naming == 'new reg':  # 6h new registration
        # if kymo_type == 'mask':
        #     folder = 'binaryMask_manual'
        #     path = data_dir + exp + '/' + folder + '/' + 'P' + str(sample)+'.tif'
        #     dtype = np.uint8
        #     print(path)
        #     return ~ tifffile.imread(path, dtype=dtype)
        if kymo_type == 'intensity':
            folder = 'intensityKymo'
            file = ''
            dtype = np.uint16
        if kymo_type == 'detrended':
            folder = 'detrendedkymo'
            file = 'detrended_'
            dtype = np.uint32
        if kymo_type == 'TL':
            folder = 'tlKymo'
            file = ''
            dtype = np.uint16
        if kymo_type == 'phase':
            folder = 'phasekymo'
            file = 'phase_'
            dtype = np.uint32
        if kymo_type == 'period':
            folder = 'periodkymo'
            file = 'period_'
            dtype = np.uint32

    if naming == 'long':  # long experiments
        if kymo_type == 'intensity':
            folder = 'intensityKymo'
            file = ''
            dtype = np.uint16
        if kymo_type == 'detrended':
            folder = 'detrendedKymo'
            file = 'detrended_'
        if kymo_type == 'phase':
            folder = 'phaseKymo'
            file = 'phase_'
            dtype = np.uint16
        if kymo_type == 'period':
            folder = 'periodKymo'
            file = 'period_'
            dtype = np.uint16
        if kymo_type == 'mask':
            folder = 'binaryKymo'
            file = ''
            dtype = np.uint8

    if naming == 'hilbert':  # Hilbert (6h)
        if kymo_type == 'mask':
            # data_dir = 'raw_data_kymos/'
            folder = 'binaryKymo/'
            file = ''
        if kymo_type == 'intensity':
            folder = 'intensityKymo/'
            file = ''
        if kymo_type == 'detrended':
            folder = 'detrendedkymo_posteriorRegistered'
            file = 'detrendedPosterior_'
        if (kymo_type ==
                'period'):
            folder = 'hilbert_periodKymo'
            file = 'htPeriod_'
        if kymo_type == 'phase':
            folder  = 'hilbert_phaseKymo'
            file = 'htPhase_'

    if naming == 'prereg':
        path = data_dir + exp + '/' + folder1 + '/' + folder2 + '/' + file + date + '_P' + str(sample) + '.tif'
    elif naming == 'new reg':
        path = data_dir + exp + '/Kymos/' + folder + '/' + file + date + '_P' + str(sample) + '.tif'
    else:
        path = data_dir + exp + '/' + folder + '/' + file + date + '_P' + str(sample) + '.tif'

    # print(path) # ________ print to check if the correct file is loaded _______________________________________

    return tifffile.imread(path, dtype='float')   # dtype = dtype

#
# temp_20210604 = pd.read_csv(data_dir+'20210604_temperature.csv', names=('mins', 'time', 'left', 'right', 'diff l/r', 'diff l', 'diff r'))
# temp_20210604['avg temp'] = (temp_20210604['left'] + temp_20210604['right'])/2.
# temp_20210604.head(3)
#
# temp_20210605 = pd.read_csv(data_dir+'20210605_temperature.csv', names=('mins', 'time', 'left', 'right', 'diff l/r', 'diff l', 'diff r'))
# temp_20210605['avg temp'] = (temp_20210605['left'] + temp_20210605['right'])/2.
# temp_20210605.head(3)

