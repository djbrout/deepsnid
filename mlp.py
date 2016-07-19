from theano import tensor
import numpy as np

x = tensor.matrix('features')

num_hidden_nodes = 50000
input_dim = 70

from blocks.bricks import Linear, Rectifier, Softmax
input_to_hidden = Linear(name='input_to_hidden',
                         input_dim=input_dim, output_dim=num_hidden_nodes)
h = Rectifier().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output',
                         input_dim=num_hidden_nodes, output_dim=2)
y_hat = Softmax().apply(hidden_to_output.apply(h))

y = tensor.lmatrix('targets')
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)


from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
cg = ComputationGraph(cost)
W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
L1,L2 = 0.05, 0.05
cost = cost + L1 * (W1 ** 2).sum() + L2 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

from blocks.bricks import MLP
mlp = MLP(activations=[Rectifier(), Softmax()], dims=[input_dim,num_hidden_nodes, 2]).apply(x)
W1.name = 'W1'


from blocks.initialization import IsotropicGaussian, Constant
hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.weights_init = hidden_to_output.weights_init
hidden_to_output.biases_init = Constant(0)
input_to_hidden.biases_init = hidden_to_output.biases_init
input_to_hidden.initialize()
hidden_to_output.initialize()


#TRAINING DATA
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten


from fuel.datasets.hdf5 import H5PYDataset
train_set = H5PYDataset('ddl_sncc_known.hdf5', which_sets=('train',))
test_set = H5PYDataset('ddl_sncc_known.hdf5', which_sets=('test',))
testia_set = H5PYDataset('ddl_sncc_known.hdf5', which_sets=('test_ia',))
testnonia_set = H5PYDataset('ddl_sncc_known.hdf5', which_sets=('test_nonia',))

data_stream = Flatten(DataStream.default_stream(train_set,
                                                iteration_scheme=SequentialScheme(700, batch_size=50)))
data_stream_test = Flatten(DataStream.default_stream(test_set,
                                                iteration_scheme=SequentialScheme(556, batch_size=50)))

# #MAKE A DATASTREAM OF ONLY Ias FOR EFFIC CALC
data_stream_testia = Flatten(DataStream.default_stream(testia_set,
                                                      iteration_scheme=SequentialScheme(383, batch_size=50)))
#
#MAKE A DATASTREAM OF ONLY CC FOR PURITY CALC
data_stream_testnonia = Flatten(DataStream.default_stream(testnonia_set,
                                                        iteration_scheme=SequentialScheme(173, batch_size=50)))

from blocks.algorithms import GradientDescent, Scale
algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                             step_rule=Scale(learning_rate=0.0001))


probs = hidden_to_output.apply(input_to_hidden.apply(x))
error_rate = (MisclassificationRate().apply(y.flatten(), probs).copy(name='error_rate'))
error_rate.name = 'error_rate'

# print(W1.container)
# import sys
# sys.exit()
#MONITOR TEST PERFORMANCE
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring

monitor = DataStreamMonitoring(variables=[cost, error_rate], data_stream=data_stream_test, prefix="test")
monitor2 = DataStreamMonitoring(variables=[error_rate], data_stream=data_stream_testnonia, prefix="nonia")
monitor3 = DataStreamMonitoring(variables=[error_rate], data_stream=data_stream_testia, prefix="ia")
monitor4 = TrainingDataMonitoring([error_rate], after_batch=True,prefix='train')


import dlplots
import matplotlib.pyplot as plt

#dlplots.hinton(W1.eval())
#plt.savefig('W1_before.png')

#MAIN LOOP
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks_extras.extensions.plot import Plot
from blocks_extras.extensions.predict import PredictDataStream

dobokeh = False
if not dobokeh:
    main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=[monitor,monitor2,monitor3,monitor4, FinishAfter(after_n_epochs=50000), Printing()])
else:    # bokeh-server
    main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=[FinishAfter(after_n_epochs=20000), TrainingDataMonitoring([cost,error_rate], after_batch=True),
                                 Plot('Classifier', channels=[['cost_with_regularization'],['error_rate']], after_batch=True)])

main_loop.run()

np.savez('sncc_results.npz',main_loop=main_loop)
#dlplots.hinton(W1.eval())
plt.savefig('W1_after.png')

#print(monitor.data_stream)
#print(probs.__di)
#dlplots.confusion_matrix(y.eval(),probs.eval())
#plt.savefig('confusion_matrix.png')
print('DONEEE')