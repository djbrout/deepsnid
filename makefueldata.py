import numpy as np
import sys
import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
#data1 = np.load('dldatabig_7day_bazin.npz')
data1= np.load('lcfits/smearG10+CCx1.npz')
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

#print big_data_array.shape
#raw_input()

print big_data_array.shape

big_data_array = np.delete(big_data_array,[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],axis=1)
print big_data_array.shape
#raw_input()
#print np.min(big_data_array,axis=0)

numex = 300

iac = 0
niac = 0
ia_train_index = []
nia_train_index = []
ia_test_index = []
nia_test_index = []
index = -1
for sn in truetypes:
    index += 1
    if iac > numex:
        ia_test_index.append(index)
    elif sn == 0:
        iac += 1
        ia_train_index.append(index)
        
index =-1
for sn in truetypes:
    index += 1
    if niac > numex:
        nia_test_index.append(index)
    elif sn != 0:
        #print index
        #print nia_train_index
        niac += 1
        nia_train_index.append(index)
print ia_train_index

#ia_train_index = np.array(ia_train_index)
#nia_train_index = np.array(nia_train_index)
#ia_test_index = np.array(ia_test_index)
#nia_test_index = np.array(nia_test_index)

        
for param in range(big_data_array.shape[1]):
    vals = big_data_array[:,param] 
    where = vals == -999
    big_data_array[where,param] = big_data_array[where,param]*0. + np.mean(big_data_array[~where,param])
    std = np.std(big_data_array[:, param])
    ww = np.abs(big_data_array[:, param]) > 3. * std
    big_data_array[ww, param] = np.mean(big_data_array[:, param])

    big_data_array[:,param] = big_data_array[:,param] - np.mean(big_data_array[:,param],axis=0)
    big_data_array[:, param] = big_data_array[:,param]/np.std(big_data_array[:,param])

    ww = np.abs(big_data_array[:,param]) > 3*np.std(big_data_array[:,param])
    big_data_array[ww , param] = np.mean(big_data_array[~ww,param])

    if param > 2:
        big_data_array[:, param] = big_data_array[:,param] - np.min(big_data_array[:,param]) + .1
        big_data_array[:, param] = 1./big_data_array[:, param]
    plt.clf()
    plt.hist(big_data_array[:,param],bins=50)
    plt.savefig('paramplots/param_'+str(param)+'.png')
    
print np.min(big_data_array,axis=0)
#plot histograms

#input()
numex = len(truetypes)

split = 700

indices = np.arange(numex)
#np.random.shuffle(indices)

print(big_data_array.shape)

traini = np.concatenate((ia_train_index,nia_train_index))#,dtype=np.uint8)
testi = np.concatenate((ia_test_index,nia_test_index))#,dtype=np.uint8)

np.random.shuffle(traini)
np.random.shuffle(testi)
print traini.shape

#traini = np.array(traini,dtype=np.uint8)
#testi = np.array(testi,dtype=np.uint8)

train_features = big_data_array[traini, :]
train_targets = truetypes[traini]
#
test_features = big_data_array[testi, :]
test_targets = truetypes[testi]

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

print('len(train)',len(train_targets),traini.shape,'len(test)',len(test_targets),testi.shape)
print('testia',len(test_targets_ia),'testnonia',len(test_targets_nonia))
import h5py
f = h5py.File('fueldata/ddl_smearG10+CCx1.hdf5', mode='w')
features = f.create_dataset('features', (len(train_features) + len(test_features) + len(test_features_ia) + len(test_features_nonia), big_data_array.shape[1]), dtype='float32')
targets = f.create_dataset('targets', (len(train_features) + len(test_features) + len(test_targets_ia) + len(test_targets_nonia), 1), dtype='uint8')

#features = f.create_dataset('features', (numex , big_data_array.shape[1]), dtype='float32')
#targets = f.create_dataset('targets', (numex , 1), dtype='uint8')

#features[...] = np.vstack([train_features, test_features])#, test_features_ia, test_features_nonia])
features[...] = np.vstack([train_features, test_features, test_features_ia, test_features_nonia])

train_targetsf = np.zeros((len(train_targets),1))
test_targetsf = np.zeros((len(test_targets),1))
test_targetsf_ia = np.zeros((len(test_targets_ia),1))
test_targetsf_nonia = np.zeros((len(test_targets_nonia),1))


train_targetsf[:,0] = train_targets
test_targetsf[:,0] = test_targets
test_targetsf_ia[:,0] = test_targets_ia
test_targetsf_nonia[:,0] = test_targets_nonia

#targets[...] = np.vstack([train_targetsf, test_targetsf])#, test_targetsf_ia, test_targetsf_nonia])
targets[...] = np.vstack([train_targetsf, test_targetsf, test_targetsf_ia, test_targetsf_nonia])

#print('yo',len(targets), numex + len(test_targets_ia) + len(test_targets_nonia))
#print('yo',len(features),numex + len(test_features_ia) + len(test_features_nonia))


features.dims[0].label = 'batch'
features.dims[1].label = 'feature'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'index'

print 'train targets',len(train_targets)
# print('ia targets',len(test_targets_ia))
# print('nonia targets',len(test_targets_nonia))
numex=len(train_targets)+len(test_targets)
from fuel.datasets.hdf5 import H5PYDataset
split_dict = {'train': {'features': (0, len(train_targets)),
                        'targets': (0, len(train_targets))},
              'test': {'features': (len(train_targets), numex),
                       'targets': (len(train_targets), numex)},
              'test_ia': {'features': (numex, numex+len(test_features_ia)),
                        'targets': (numex, numex+len(test_targets_ia))},
              'test_nonia': {'features': (numex + len(test_features_ia), numex + len(test_features_ia) + len(test_features_nonia)),
                        'targets': (numex + len(test_targets_ia), numex + len(test_targets_ia) + len(test_targets_nonia))}
              }
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
print(split)
f.flush()
f.close()
