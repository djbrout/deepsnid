import numpy as np
import sys

#data1 = np.load('dldatabig_7day_bazin.npz')
data1= np.load('sncc_known2.npz')
#data2 = np.load('dldatabig_bettercadence2.npz')

#truetypes = np.hstack((data1['truetypes'],data2['truetypes']))
#big_data_array = np.vstack((data1['big_data_array'],data2['big_data_array']))

truetypes = data1['truetypes']
big_data_array = data1['big_data_array']
#big_data_array[:,4:24] = big_data_array[:,4:24]/np.max(big_data_array[:,4:24],axis=0)
#print(big_data_array.shape)
big_data_array = big_data_array[:,:]
#print(np.max(big_data_array,axis=0),np.min(big_data_array,axis=0),np.median(big_data_array,axis=0))
big_data_array[:,:3] = big_data_array[:,:3]/np.max(big_data_array[:,:3],axis=0)
#plot histograms

#input()
numex = len(truetypes)

split = 700

indices = np.arange(numex)
np.random.shuffle(indices)


#big_data_array = np.delete(big_data_array,5,axis=1)

# for i,t in enumerate(data['truetypes']):
#     big_data_array[i,5] = t
print(big_data_array.shape)
#print(data['truetypes'])

train_features = big_data_array[indices[:split], :]
train_targets = truetypes[indices[:split]]
#
test_features = big_data_array[indices[split:], :]
test_targets = truetypes[indices[split:]]

# np.savez('nndata_7daycad.npz',train_features=train_features,train_targets=train_targets,
#          test_features=test_features,test_targets=test_targets)

# train_features = big_data_array[:split, :]
# train_targets = data['truetypes'][:split]
#
# test_features = big_data_array[split:, :]
# test_targets = data['truetypes'][split:]
# print(test_targets)

test_features_ia = test_features[test_targets == 0]
test_targets_ia = test_targets[test_targets == 0]

test_features_nonia = test_features[test_targets == 1]
test_targets_nonia = test_targets[test_targets == 1]

#print('test_features_ia',test_features_ia[:,3],'test_targest_ia',test_targets_ia)


print('len(train)',len(train_targets),'len(test)',len(test_targets))
print('testia',len(test_targets_ia),'testnonia',len(test_targets_nonia))
import h5py
f = h5py.File('ddl_sncc_known.hdf5', mode='w')
features = f.create_dataset('features', (numex + len(test_features_ia) + len(test_features_nonia), big_data_array.shape[1]), dtype='float32')
targets = f.create_dataset('targets', (numex + len(test_targets_ia) + len(test_targets_nonia), 1), dtype='uint8')

#features = f.create_dataset('features', (numex , big_data_array.shape[1]), dtype='float32')
#targets = f.create_dataset('targets', (numex , 1), dtype='uint8')

#features[...] = np.vstack([train_features, test_features])#, test_features_ia, test_features_nonia])
features[...] = np.vstack([train_features, test_features, test_features_ia, test_features_nonia])

train_targetsf = np.zeros((split,1))
test_targetsf = np.zeros((numex-split,1))
test_targetsf_ia = np.zeros((len(test_targets_ia),1))
test_targetsf_nonia = np.zeros((len(test_targets_nonia),1))


train_targetsf[:,0] = train_targets
test_targetsf[:,0] = test_targets
test_targetsf_ia[:,0] = test_targets_ia
test_targetsf_nonia[:,0] = test_targets_nonia

#targets[...] = np.vstack([train_targetsf, test_targetsf])#, test_targetsf_ia, test_targetsf_nonia])
targets[...] = np.vstack([train_targetsf, test_targetsf, test_targetsf_ia, test_targetsf_nonia])

print('yo',len(targets), numex + len(test_targets_ia) + len(test_targets_nonia))
print('yo',len(features),numex + len(test_features_ia) + len(test_features_nonia))


features.dims[0].label = 'batch'
features.dims[1].label = 'feature'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'index'


# print('ia targets',len(test_targets_ia))
# print('nonia targets',len(test_targets_nonia))

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {'train': {'features': (0, split),
                        'targets': (0, split)},
              'test': {'features': (split, numex),
                       'targets': (split, numex)},
              'test_ia': {'features': (numex, numex+len(test_features_ia)),
                        'targets': (numex, numex+len(test_targets_ia))},
              'test_nonia': {'features': (numex + len(test_features_ia), numex + len(test_features_ia) + len(test_features_nonia)),
                        'targets': (numex + len(test_targets_ia), numex + len(test_targets_ia) + len(test_targets_nonia))}
              }
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
print(split)
f.flush()
f.close()