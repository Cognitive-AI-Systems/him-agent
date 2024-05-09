#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import socket
import time
import json

EPS = 1e-24
INT_TYPE = "int64"
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 1e-24


def softmax(x, beta=1.0):
    e_x = np.exp(beta * (x - x.max()))
    return e_x / e_x.sum()


def sigmoid(x):
    return 1/(1+np.exp(-x))


def normalize(x, default_values=None):
    norm_x = x.copy()
    norm = x.sum(axis=-1)
    mask = norm == 0

    if default_values is None:
        default_values = np.ones_like(x)

    norm_x[mask] = default_values[mask]
    norm[mask] = norm_x[mask].sum(axis=-1)
    return norm_x / norm.reshape((-1, 1))


def sample_categorical_variables(probs, rng: np.random.Generator):
    assert np.allclose(probs.sum(axis=-1), 1)

    gammas = rng.uniform(size=probs.shape[0]).reshape((-1, 1))

    dist = np.cumsum(probs, axis=-1)
    dist[:, -1] = 1.0

    ubounds = dist
    lbounds = np.zeros_like(dist)
    lbounds[:, 1:] = dist[:, :-1]

    cond = (gammas >= lbounds) & (gammas < ubounds)

    states = np.zeros_like(probs) + np.arange(probs.shape[1])

    samples = states[cond].astype(UINT_DTYPE)

    return samples


def get_data(connection: socket):
    try:
        data = None
        while not data:
            data = connection.recv(4)
            time.sleep(0.000001)

        length = int.from_bytes(data, "little")
        string = ""
        while (
                len(string) != length
        ):  # TODO: refactor as string concatenation could be slow
            string += connection.recv(length).decode()

        return string
    except socket.timeout as e:
        print("timed out", e)

    return None


def send_string(string, connection: socket):
    message = len(string).to_bytes(4, "little") + bytes(string.encode())
    connection.sendall(message)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
