import brian2 as b2
import numpy as np


class Lifq_3d:
    """
    Attributes : 
    state : Is recording the state ( membrane potential) of each neuron during the simulation
    spike : Is the recording of the firing time of each neuron during the simulation
    matrix : matrix is the matrix associated to the entry signal in a type understandable by brian2
    reconstr_array : is the matrix containing the reconstructed signal after passing through the LIF Quantizer
    """
    def __init__(self):
        self.state = None
        self.spike = None
        self.matrix = None
        self.reconstr_array = None

    def _reformat_video_frame_matrix(self, video_frame, time, save_matrix,  logger):
        reformated_matrix = []
        if logger:
            print("Matrix construction, it may take some time")

        for frame in video_frame:
            reformated_matrix.append(
                self.create_time_matrix(frame, time/2, time))
        if save_matrix:
            np.save('video_frame.npy', np.asarray(
                reformated_matrix, dtype=np.float64))
        if logger:
            print("Matrix is now complete, moving to neuron simulation")
        return np.asarray(reformated_matrix, dtype=np.float64)

    def _create_time_matrix(self, matrix, time, simulation_time):
        big_matrix = np.empty(np.int64(np.ceil(simulation_time/time)), )
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                temp = np.hstack(c for c in np.full(
                    (np.int64(np.ceil(simulation_time/time)), 1), matrix[i][j]/255))
                big_matrix = np.vstack((big_matrix, temp))
        big_matrix = np.delete(big_matrix, 0, 0)
        return big_matrix

    def _simulate_LIF_neuron(self, input_current, N, simulation_time, v_rest,
                            v_reset, firing_threshold, membrane_resistance, membrane_time_scale,
                            abs_refractory_period):
        # Construct the network just once
        # LIF neuron using Brian2 library
        self.spike = []
        eqs = """
            dv/dt =
            ( -(v-v_rest) + membrane_resistance * input_current(t, i) ) / membrane_time_scale : volt (unless refractory)"""

        neuron = b2.NeuronGroup(
            N, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
            refractory=abs_refractory_period, method="euler")
        neuron.v = v_rest  # set initial value

        spike_monitor = b2.SpikeMonitor(neuron)
        # Store the current state of the network
        b2.store()
        for frame in self.matrix:
            # Load the new frame
            input_current = b2.TimedArray(np.transpose(
                frame) * b2.mA, dt=simulation_time/2)
            # print(spike_monitor.count)

            # Restore the original state of the network
            b2.restore()
            # Run it with the new frame
            b2.run(simulation_time)
            self.spike.append(np.asarray(spike_monitor.count))

    def _decode_all(self, video_frame, simulation_time, firing_threshold, membrane_resistance, membrane_time_scale):
        self.reconstr_array = []
        for spike_count in self.spike:
            self.reconstr_array.append(self._decode_single_frame(
                spike_count, video_frame, simulation_time, firing_threshold, membrane_resistance, membrane_time_scale))

    def _decode_single_frame(self, spike_count, video_frame, simulation_time, firing_threshold, membrane_resistance, membrane_time_scale):

        dict_u_hat = dict()
        for values in np.unique(spike_count):
            if values == 0:
                dict_u_hat[values] = 0
            else:
                d_u_hat = simulation_time/values
                dict_u_hat[values] = (
                    ((firing_threshold/(1-np.exp(-(d_u_hat/membrane_time_scale))))*(1/membrane_resistance)) / b2.mA)*255
        temp_array = np.ndarray((len(video_frame[0]), len(video_frame[0][0])))
        for i in range(len(spike_count)):
            temp_array[np.int64(np.floor(i/len(video_frame[0])))][i % len(
                video_frame[0][0])] = np.int64(np.floor(dict_u_hat[spike_count[i]]))
        return temp_array

    def fit(self, X, simulation_time=66 * b2.ms, v_rest=0 * b2.mV,
            v_reset=0 * b2.mV, firing_threshold=0.09 * b2.mV,
            membrane_time_scale=7 * b2.ms, membrane_resistance=550 * b2.mohm,
            abs_refractory_period=0 * b2.ms, logger=False, save_matrix=False,
            load_matrix=False):
            """
            Apply the lif quantizer to the data
            parameters :
            X : numpy array, is the input signal
            simulation_time : time during which the simulation will be performed must be a of type (second)
            firing threshold :  threshold at which the neuron will spike must be of type (volt)
            membrane_time_scale : must be of type (second)
            membrane_resistance : must be of type (ohm)
            abs_refractory_period : period after a spike in which the neuron will do nothing, must be type (second)
            logger : if set to True will enablethe brianlogger
            save_matrix : if set to True, will save the matrix computed from the input data,  it's usefull if you plan on
            testing multiple setup for the same data input, 
            load_matrix : is set to True, the matrix will not be re calculated
             instead it will load the matrix calculated beforehand, it saves a lot of computing time
        """
        if not isinstance(X, np.ndarray):
            X  = np.asarray(X)
        assert X.ndim > 2, "Please provide an array with at least three dimension"
        assert X.shape[1] >= X.shape[
            2], "Please resize the image to a format (n * m) where n >= m"
       # if X
        if X.ndim == 4:
            # Convert to grayscale
            X = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        N = X.shape[1] * X.shape[2]
        if load_matrix:
            try:
                self.matrix = np.load('video_frame.npy')
            except FileNotFoundError:
                print("Could not load matrix")
                self.matrix = self._reformat_video_frame_matrix(
                X, simulation_time, save_matrix, logger)


        else:
            self.matrix = self._reformat_video_frame_matrix(
                X, simulation_time, save_matrix, logger)

        if logger:
            b2.BrianLogger.log_level_debug()
        self._simulate_LIF_neuron(self.matrix, N, simulation_time, v_rest,
                                 v_reset, firing_threshold, membrane_resistance, membrane_time_scale,
                                 abs_refractory_period)

        self._decode_all(X, simulation_time, firing_threshold,
                        membrane_resistance, membrane_time_scale)

    def getSpike(self):
        if not isinstance(self.spike, type(None)):
            return self.spike
        else:
            raise AttributeError("You cannot call getSpike before fit")

    def getDecodedSignal(self):
        if not isinstance(self.reconstr_array, type(None)):
            return self.reconstr_array
        else:
            raise AttributeError("You cannot call getDecodedSignal before fit")
