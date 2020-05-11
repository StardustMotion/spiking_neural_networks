import brian2 as b2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim



class Lifq_2d:

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

    def _simulate_LIF_neuron(self, input_current, N, simulation_time, v_rest,
                            v_reset, firing_threshold, membrane_resistance, membrane_time_scale,
                            abs_refractory_period):
        # differential equation of Leaky Integrate-and-Fire model
        eqs = """
        dv/dt =
        ( -(v-v_rest) + membrane_resistance * input_current(t, i) ) / membrane_time_scale : volt (unless refractory)"""

        # LIF neuron using Brian2 library
        neuron = b2.NeuronGroup(
            N, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
            refractory=abs_refractory_period, method="euler")
        neuron.v = v_rest  # set initial value

        # monitoring membrane potential of neuron and injecting current
        state_monitor = b2.StateMonitor(neuron, ["v"], record=True)
        spike_monitor = b2.SpikeMonitor(neuron)
        # run the simulation
        b2.run(simulation_time)
        return state_monitor, spike_monitor


    def _create_time_matrix(self, matrix, time, simulation_time, is_pixel):
        big_matrix = np.empty(np.int64(np.ceil(simulation_time/time)), )
        if is_pixel:
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    temp = np.hstack(c for c in np.full((np.int64(np.ceil(simulation_time/time)), 1), matrix[i][j]/255 ))
                    big_matrix = np.vstack((big_matrix, temp))
        else:
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    temp = np.hstack(c for c in np.full((np.int64(np.ceil(simulation_time/time)), 1), matrix[i][j] ))
                    big_matrix = np.vstack((big_matrix, temp))
        big_matrix = np.delete(big_matrix, 0, 0)
        return b2.TimedArray(np.transpose(big_matrix) * b2.mA, dt = time)

    def _proba(self, i, spike_count):
        unique, counts = np.unique(spike_count, return_counts=True)
        dict_temp = dict(zip(unique, counts))
        return dict_temp[spike_count[i]]/len(spike_count)


    def _compute_entropy(self, spike_count):
        entropy = 0
        for i in np.unique(spike_count, return_index = True)[1]:
            entropy += self._proba(i,  spike_count)*np.log(self._proba(i, spike_count))
        return -entropy

    def _decode2D(self, spike_count, firing_threshold,
               membrane_time_scale, membrane_resistance, simulation_time, shape, is_pixel):
        dict_u_hat = dict()
        if is_pixel:
            for values in np.unique(spike_count):
                if values == 0:
                    dict_u_hat[values] = 0
                else : 
                    d_u_hat = simulation_time/values
                    dict_u_hat[values] = (((firing_threshold/(1-np.exp(-(d_u_hat/membrane_time_scale))))*(1/membrane_resistance)) /b2.mA)*255
        else:
            for values in np.unique(spike_count):
                if values == 0:
                    dict_u_hat[values] = 0
                else : 
                    d_u_hat = simulation_time/values
                    dict_u_hat[values] = (((firing_threshold/(1-np.exp(-(d_u_hat/membrane_time_scale))))*(1/membrane_resistance)) /b2.mA)

        temp = np.ndarray((shape[0],shape[1]))
        if is_pixel : 
            for i in range(len(spike_count)):
                temp[np.int64(np.floor(i/shape[0]))][i%shape[1]] = np.int64(np.floor(dict_u_hat[spike_count[i]]))
        else:
            for i in range(len(spike_count)):
                temp[np.int64(np.floor(i/shape[0]))][i%shape[1]] = dict_u_hat[spike_count[i]]

        return temp

    def fit(self, X, simulation_time=66 * b2.ms, v_rest=0 * b2.mV,
            v_reset=0 * b2.mV, firing_threshold=0.09 * b2.mV,
            membrane_time_scale=7 * b2.ms, membrane_resistance=550 * b2.mohm, abs_refractory_period=0 * b2.ms, logger=False, is_pixel=True):
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
            is_pixel : if is_pixel is don't change, it will expect data to be in range (0,255) and will scale them to [0, 1], if  you want to use your own transformation, set is_pixel to false


            if is_pixel is set to false and the data you provide is not in [0, 1] be sure to change the parameters for the lif simulation
        """
        if not isinstance(X, np.ndarray):
            X  = np.asarray(X)
        assert X.ndim == 2, "Dimmention Error, input must be of dimmention 2 not {}".format(X.ndim)
        assert X.shape[0] >= X.shape[1], "Please resize the image to a format (n * m) where n >= m" 
        if logger:
            b2.BrianLogger.log_level_debug()

        self.matrix = self._create_time_matrix(
            X, simulation_time, simulation_time, is_pixel)
        self.state, self.spike = self._simulate_LIF_neuron(self.matrix, X.shape[0]*X.shape[1], simulation_time, v_rest,
                                                          v_reset, firing_threshold, membrane_resistance,
                                                          membrane_time_scale, abs_refractory_period)
        self.reconstr_array = self._decode2D(
            self.spike.count,
            firing_threshold,
            membrane_time_scale,
            membrane_resistance,
            simulation_time,
            X.shape, is_pixel)

    def getSpike(self):
        """
        return the spike recordings, don't use getSpike before fit
        """
        if not isinstance(self.spike, type(None)):
            return self.spike
        else:
            raise AttributeError("You cannot call getSpike before fit")

    def getState(self):
        """
        return the state recordings, don't use getState before fit
        """
        if not isinstance(self.state, type(None)):
            return self.state
        else:
            raise AttributeError("You cannot call getState before fit")

    def getDecodedSignal(self):
        """
        return the signal decoded from the spike train, don't use getDecodedSignal before fit
        """
        if not isinstance(self.reconstr_array, type(None)):
            return self.reconstr_array
        else:
            raise AttributeError("You cannot call getDecodedSignal before fit")

    def  getEntropy(self):
        """
        Return the Entropy  of the signal, don't use getEntropy before fit
        """
        if not isinstance(self.spike, type(None)):
            return self._compute_entropy(self.spike.count)
        else:
            raise AttributeError("You cannot call getEntropy before fit")





            