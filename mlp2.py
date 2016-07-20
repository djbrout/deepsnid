#!/usr/bin/env python

import logging
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import MLP, Tanh, Softmax, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.roles import WEIGHT

try:
    from blocks_extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


def main(save_to, num_epochs):
    mlp = MLP([Tanh(), Softmax()], [64, 10000, 2],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    #mlp = MLP([Rectifier(), Softmax()], [64, 10000, 2],
    #          weights_init=IsotropicGaussian(0.01),
    #          biases_init=Constant(0))
    mlp.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    probs = mlp.apply(x)
    cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
    error_rate = MisclassificationRate().apply(y.flatten(), probs)

    cg = ComputationGraph([cost])
    W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + .00005 * (W1 ** 2).sum() + .00005 * (W2 ** 2).sum()
    cost.name = 'final_cost'

    #mnist_train = MNIST(("train",))
    #mnist_test = MNIST(("test",))
    from fuel.datasets.hdf5 import H5PYDataset

    train_set = H5PYDataset('./fueldata/ddl_smearG10+CCx1.hdf5', which_sets=('train',))
    test_set = H5PYDataset('./fueldata/ddl_smearG10+CCx1.hdf5', which_sets=('test',))
    test_iaset = H5PYDataset('./fueldata/ddl_smearG10+CCx1.hdf5', which_sets=('test_ia',))
    test_niaset = H5PYDataset('./fueldata/ddl_smearG10+CCx1.hdf5', which_sets=('test_nonia',))

    train = DataStream.default_stream(train_set,
                              iteration_scheme=SequentialScheme(602, batch_size=50))
    test = DataStream.default_stream(test_set,
                              iteration_scheme=SequentialScheme(829, batch_size=50))
    testia = DataStream.default_stream(test_iaset,
                              iteration_scheme=SequentialScheme(549, batch_size=50))
    testnia = DataStream.default_stream(test_niaset,
                              iteration_scheme=SequentialScheme(280, batch_size=50))

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=0.05))

    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      Flatten(test),
                      prefix="test"),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      Flatten(testia),
                      prefix="Ia Purity"),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      Flatten(testnia),
                      prefix="CC Purtiy"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  Printing()]

    '''if BLOCKS_EXTRAS_AVAILABLE:
        extensions.append(Plot(
            'Example',
            channels=[
                ['test_final_cost',
                 'test_misclassificationrate_apply_error_rate'],
                ['train_total_gradient_norm']]))
    '''
    main_loop = MainLoop(
        algorithm,
        Flatten(train),
        model=Model(cost),
        extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=30000,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="results/tanh_10k.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs)
