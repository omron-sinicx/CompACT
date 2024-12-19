import h5py

filename_hdf = '/root/osx-ur/dependencies/comp-act/datasets/sim_bimanual_wiping_compliance/episode_0.hdf5'


def h5_tree(val, pre=''):
    if val.attrs:
        for k in val.attrs.keys():
            print("Attr: ", k, '=', val.attrs[k])
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + f' ({val.shape})')
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + f' ({val.shape})')
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


with h5py.File(filename_hdf, 'r') as hf:
    print(filename_hdf)
    print(hf)
    h5_tree(hf)
