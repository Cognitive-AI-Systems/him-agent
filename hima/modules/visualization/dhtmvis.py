#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import socket
import time
import json
import atexit

HOST = "127.0.0.1"
PORT = 5555
TIMEOUT = 600


class DHTMVisualizer:
    def __init__(self, host=HOST, port=PORT, main_window_size=(200, 50), time_constant=50):
        self.host = host
        self.port = port
        self.main_window_size = main_window_size
        self.time_constant = time_constant

        self.data = None
        self.n_steps = 0
        self.phase = 'unknown'

        # create figures
        self.fig_messages = plt.figure('messages')
        self.messages = self.fig_messages.subplot_mosaic(
            [
                ['external', '.', '.'],
                ['context', 'prediction', 'internal'],
                ['.', 'obs_states_prediction', 'obs_states']
            ],
            height_ratios=[0.25, 1, 0.25],
            per_subplot_kw={
                'external': {'title': 'external'},
                'context': {'title': 'context'},
                'prediction': {'title': 'prediction'},
                'internal': {'title': 'internal'},
                'obs_states_prediction': {'title': 'obs_states_prediction'},
                'obs_states': {'title': 'obs_states'}
            }
        )
        self.fig_segments = plt.figure('segments')
        self.segments = self.fig_segments.subplot_mosaic(
            [
                ['.', 'external_fields_of_new', '.'],
                ['total_per_cell', 'context_fields_of_new', 'cells_to_grow_new'],
            ],
            height_ratios=[0.25, 1],
            per_subplot_kw={
                'external_fields_of_new': {'title': 'external_fields'},
                'total_per_cell': {'title': 'total_per_cell'},
                'context_fields_of_new': {'title': 'context_fields'},
                'cells_to_grow_new': {'title': 'cells_to_grow_new'}
            }
        )

        self.fig_messages.canvas.mpl_connect('key_press_event', self.step)

        # create server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            while True:
                try:
                    s.bind((self.host, self.port))
                    break

                except OSError:
                    self.port += 1
                    if self.port > 49151:
                        self.port = 1024

            s.listen(1)
            self.connection, addr = s.accept()
            print(
                f'connected {addr}'
            )

        data = self._get_json_dict()
        if data['type'] == 'hello':
            self._send_string('dhtm')
        else:
            self.close()

        atexit.register(self.close)
        plt.show()

    def step(self, event):
        if event.key == 'down':
            # send request
            self._send_string('step')
            # get data
            msg = self._get_json_dict()
            # read phase
            while msg['type'] != 'state':
                if msg['type'] == 'phase':
                    if msg['phase'] != self.phase:
                        self.n_steps = 0
                    self.phase = msg['phase']
                msg = self._get_json_dict()
            self.data = self._convert_to_numpy(msg)
            # draw data
            self.fig_messages.suptitle(f'phase: {self.phase}; step {self.n_steps}', fontsize=14)
            self.fig_segments.suptitle(f'phase: {self.phase}; step {self.n_steps}', fontsize=14)
            self._draw()
            self.n_steps += 1
        elif event.key == 'up':
            self._clear_axes()
            self._send_string('skip')
            self.fig_messages.suptitle(f'phase: {self.phase}; step {self.n_steps}', fontsize=14)
            self.fig_segments.suptitle(f'phase: {self.phase}; step {self.n_steps}', fontsize=14)
            plt.show()
            self.n_steps += 1
        elif event.key == 'x':
            self.close()
        elif event.key == 'right':
            while True:
                self._send_string('skip')
                msg = self._get_json_dict()
                if msg['type'] == 'phase':
                    if msg['phase'] != self.phase:
                        self.phase = msg['phase']
                        self.n_steps = 0
                        self._clear_axes()
                        self.fig_messages.suptitle(f'phase: {self.phase}; step {self.n_steps}', fontsize=14)
                        self.fig_segments.suptitle(f'phase: {self.phase}; step {self.n_steps}', fontsize=14)
                        plt.show()
                        break

    def _draw(self):
        # update figures
        self._clear_axes()
        self._update_messages()
        self._update_segments()

        plt.show()

    def _update_messages(self):
        sns.heatmap(
            self.data['external_messages'],
            ax=self.messages['external'],
            cbar=False,
            annot=True
        )
        sns.heatmap(
            self.data['context_messages'],
            ax=self.messages['context'],
            annot=True,
            cbar=False
        )

        if self.data['prediction_cells'] is not None:
            sns.heatmap(
                self.data['prediction_cells'],
                ax=self.messages['prediction'],
                annot=True,
                cbar=False
            )
        sns.heatmap(
            self.data['internal_messages'],
            ax=self.messages['internal'],
            annot=True,
            cbar=False
        )
        sns.heatmap(
            self.data['prediction_columns'],
            ax=self.messages['obs_states_prediction'],
            annot=True,
            cbar=False
        )

        sns.heatmap(
            self.data['observation_messages'],
            ax=self.messages['obs_states'],
            annot=True,
            cbar=False
        )

    def _update_segments(self):
        sns.heatmap(
            self.data['segments_per_cell'],
            ax=self.segments['total_per_cell'],
            cbar=False,
            annot=True
        )

        if 'cells_for_new_segments' in self.data:
            sns.heatmap(
                self.data['cells_for_new_segments'],
                ax=self.segments['cells_to_grow_new'],
                cbar=False,
                annot=True
            )

            sns.heatmap(
                self.data['context_fields_for_new_segments'],
                ax=self.segments['context_fields_of_new'],
                cbar=False,
                annot=True
            )
            sns.heatmap(
                self.data['external_fields_for_new_segments'],
                ax=self.segments['external_fields_of_new'],
                cbar=False,
                annot=True
            )

    def _clear_axes(self):
        for title, ax in self.messages.items():
            ax.clear()

        for title, ax in self.segments.items():
            ax.clear()

    def _send_as_json(self, dictionary):
        message_json = json.dumps(dictionary)
        self._send_string(message_json)

    def _get_json_dict(self):
        data = self._get_data()
        return json.loads(data)

    def _send_string(self, string):
        message = len(string).to_bytes(4, "little") + bytes(string.encode())
        self.connection.sendall(message)

    def _get_data(self):
        try:
            data = None
            while not data:
                data = self.connection.recv(4)
                time.sleep(0.000001)

            length = int.from_bytes(data, "little")
            string = ""
            while (
                    len(string) != length
            ):  # TODO: refactor as string concatenation could be slow
                string += self.connection.recv(length).decode()

            return string
        except socket.timeout as e:
            print("dhtm timed out", e)

        return None

    def close(self):
        plt.close('all')
        self._send_string('close')
        print("close message sent")
        time.sleep(1.0)
        self.connection.close()
        try:
            atexit.unregister(self.close)
        except Exception as e:
            print("exception unregistering close method", e)

    @staticmethod
    def _convert_to_numpy(data_dict):
        for key, value in data_dict.items():
            data_dict[key] = np.asarray(value)
        return data_dict


if __name__ == '__main__':
    vis = DHTMVisualizer()

