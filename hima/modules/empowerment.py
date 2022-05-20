#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import TemporalMemory
from itertools import product
import json
from hima.common.sdr import SparseSdr

EPS = 1e-12


class Memory:
    """
    The Memory object saves SDR representations of states and clusterizes them using the similarity measure.
    The SDR representation must have fixed sparsity of active cells for correct working.

    Parameters
    ----------
    size : int
        The size is the size of SDR representations, which are stored
    threshold: float
        The threshold is used to determine then it's necessary to create a new cluster.

    Attributes
    ----------
    size: int
        It stores size argument.
    kernels : np.array
        This is the list of created clusters representations in dence form. It contains information about frequency of
        cell's activity (for each cluster) during working. Its shape: (number of clusters, size).
    norms: np.array
        This is the list of representations amount for each cluster. Its shape: (munber of clusters, 1)
    threshold: float
        It stores threshold argument.
    """

    def __init__(self, size, threshold=0.5):
        self.kernels = None
        self.norms = None
        self.threshold = threshold
        self.size = size

    @property
    def number_of_clusters(self):
        if (self.kernels is not None) and (self.kernels.ndim == 2):
            return self.kernels.shape[0]
        else:
            return 0

    def add(self, state):
        """ Add a new SDR representation (store and clusterize).

        Parameters
        ----------
        state: np.array
            This is the SDR representation (sparse), that we want to store ande clusterize with other stored SDRs.

        Returns
        -------
        """
        state_dense = np.zeros(self.size)
        state_dense[state] = 1
        sims = self.similarity(state_dense)
        if np.sum(sims > self.threshold) == 0:
            if self.kernels is None:
                self.kernels = state_dense.reshape((1, -1))
                self.norms = np.array([[1]])
            else:
                self.kernels = np.vstack((self.kernels, state_dense))
                self.norms = np.vstack((self.norms, [1]))
        else:
            self.kernels[np.argmax(sims)] += state_dense
            self.norms[np.argmax(sims)] += 1

    def similarity(self, state):
        """This function evaluate similarity measure between stored clusters and new state.

        Parameters
        ----------
        state: np.array
            The sparse representation of the state to be compared.

        Returns
        -------
        similarities: np.array
            The similarity measures for given state. If the Memory object don't have any saved clusters, then the empty
            array is returned, else returned array contained similarities between the state and each cluster.
            Its shape: (number of kernels, 1).

        """
        if self.kernels is None:
            return np.array([])
        else:
            normalised_kernels = self.kernels / self.norms
            sims = normalised_kernels @ state.T / (
                    np.sqrt(np.sum(normalised_kernels ** 2, axis=1)) * np.sqrt(state @ state.T))
            similarities = sims.T
            return similarities

    def adopted_kernels(self, sparsity):
        """This function normalises stored representations and cuts them by sparsity threshold.

        Parameters
        ----------
        sparsity: float
            The sparsity of active cells in stored SDR representations.

        Returns
        -------
        clusters_representations: np.array
            Normalised and cutted representations of each cluster. The cutting is done by choosing the most frequent
            active cells (their number is defined by sparsity) in kernels attribute. All elements of array are
            in [0, 1]. The shape is (number of clusters, 1).
        """
        data = np.copy(self.kernels)
        data[data < np.quantile(data, 1 - sparsity, axis=1).reshape((-1, 1))] = 0
        clusters_representations = data / self.norms
        return clusters_representations


class Empowerment:
    """
    The Empowerment contains all necessary things to evaluate 'empowerment' using the model
    of environment based on Temporal Memory algorithm.

    Parameters
    ----------
    seed : int
        The seed for random generator.
    encode_size : int
        The size of SDR representations which is taken by model.
    tm_config : dict
        It contains all parameters for initialisation of the TemporalMemory without the
        'columnDimensions'. 'columnDimensions' is defined inside Empowerment.
    sparsity : float
        The sparsity of SDR representations which are used in the TemporalMemory algorithm.
    memory : bool, optional
        This parameter defines will be used the Memory for saving and clustering of state
        representations or not. By default is False (doesn't use the Memory).
    similarity_threshold : float, optional
        This parameter determines the threshold for cluster creation.
        It is used then memory is True. By default: 0.6.
    evaluate : bool, optional
        This flag defines the necessity of storing some statistics to evaluate the learning process.
        By default is True.

    Attributes
    ----------
    evaluate : bool
        It stores the same parameter.
    anomalies : list
        It stores the anomaly values of TM for each time step after learning.
        Only then 'evaluate' is True.
    IoU : list
        It stores the Intersection over Union values of TM predictions and real ones for each
        time step after learning. Only then 'evaluate' is True.
    sparsity : float
        It stores the same parameter.
    tm : TemporalMemory
        It contains the TemporalMemory object.
    size : int
        It stores the 'encode_size' parameter.
    memory: Memory
        It contains the Memory object if 'memory' parameter is True, else None.
    """

    def __init__(
            self, seed: int, encode_size: int, tm_config: dict, sparsity: float,
            memory: bool = False, similarity_threshold: float = 0.6, evaluate: bool = True,
            filename: str = None):
        self.filename = filename
        if self.filename is None:
            self.evaluate = evaluate
            if evaluate:
                self.anomalies = []
                self.IoU = []
            self.sdr_0 = SDR(encode_size)
            self.sdr_1 = SDR(encode_size)
            self.sparsity = sparsity

            self.tm = TemporalMemory(
                columnDimensions=[encode_size], seed=seed, **tm_config,
            )
            self.size = self.tm.getColumnDimensions()[0]

            if memory:
                self.memory = Memory(self.tm.getColumnDimensions()[0], threshold=similarity_threshold)
            else:
                self.memory = None
        else:
            with open(self.filename) as json_file:
                self.empowerment_data = json.load(json_file)

    def eval_from_file(self, position):
        return self.empowerment_data[str(position[0])][str(position[1])]

    def eval_state(
            self, state: SparseSdr, horizon: int,
            use_segments: bool = False, use_memory: bool = False
    ) -> float:
        """This function evaluates empowerment for given state.

        Parameters
        ----------
        state : SparseSdr
            The SDR representation (sparse) of the state.
        horizon : int
            The horizon of evaluating for given state. The good value is 3.
        use_segments : bool, optional
            The flag determines using of segments instead of cells to evaluate empowerment.
            By default: False.
        use_memory : bool, optional
            The flag determines using of the Memory object. Useful only if this object was
            initialised. By default: False.

        Returns
        -------
        float
            The empowerment value (always > 0).
        """
        self.sdr_0.sparse = state
        sdr = self.sdr_0
        data = np.zeros(self.tm.getColumnDimensions()[0])
        for actions in range(horizon):
            self.tm.reset()
            self.tm.compute(sdr, learn=False)
            self.tm.activateDendrites(learn=False)
            predictiveCells = self.tm.getPredictiveCells().sparse
            predictedColumnIndices = [self.tm.columnForCell(i) for i in predictiveCells]
            sdr.sparse = np.unique(predictedColumnIndices)
        if use_segments:
            predictedColumnIndices = map(
                self.tm.columnForCell,
                map(self.tm.connections.cellForSegment, self.tm.getActiveSegments())
            )
        for i in predictedColumnIndices:
            data[i] += 1
        if self.memory is not None and use_memory:
            if (self.memory.kernels is not None) and (self.memory.kernels.size > 0):
                clusters = self.memory.adopted_kernels(self.sparsity)
                mask = clusters[:, data!=0].sum(axis=1) / (self.sparsity * self.size)
                mask = mask < self.memory.threshold
                p = np.dot(clusters, data.T) / (self.sparsity * self.size)
                p[mask] = 0
                total_p = p.sum()
                p = p / (total_p + EPS)
                empowerment = np.sum(-p * np.log(p, where=p != 0), where=p != 0)
                return empowerment
            else:
                return 0
        p = data / (data.sum() + EPS)
        empowerment = np.sum(-p * np.log(p, where=data != 0), where=data != 0)
        return empowerment

    def learn(self, state_0, state_1):
        """This function realize learning of TM.

        Parameters
        ----------
        state_0: np.array
            The SDR representation of the state (sparse form).
        state_1: np.array
            The SDR representation of the next state (sparse form).

        Returns
        -------
        """
        self.sdr_0.sparse = state_0
        self.sdr_1.sparse = state_1
        self.tm.reset()

        if self.memory is not None:
            self.memory.add(self.sdr_0.sparse)
        self.tm.compute(self.sdr_0, learn=True)

        if self.evaluate:
            self.tm.activateDendrites(learn=False)
            predictiveCells = self.tm.getPredictiveCells().sparse
            predictedColumnIndices = np.unique([self.tm.columnForCell(i) for i in predictiveCells])

        self.tm.compute(self.sdr_1, learn=True)
        if self.evaluate:
            intersection = np.intersect1d(self.sdr_1.sparse, predictedColumnIndices)
            union = np.union1d(self.sdr_1.sparse, predictedColumnIndices)

        if self.evaluate:
            self.IoU.append(len(intersection) / len(union))
            self.anomalies.append(self.tm.anomaly)
        self.tm.reset()


def real_empowerment(env, position, horizon):
    data = np.zeros(env.env.shape)

    for actions in product(range(4), repeat=horizon):
        env.env.agent.position = position
        for a in actions:
            env.act(a)

        data[env.env.agent.position] += 1
    return np.sum(-data / data.sum() * np.log(data / data.sum(), where=data != 0), where=data != 0), data

