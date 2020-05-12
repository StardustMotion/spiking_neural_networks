from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image




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
        neuron = NeuronGroup(
            N, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
            refractory=abs_refractory_period, method="euler")
        neuron.v = v_rest  # set initial value

        # monitoring membrane potential of neuron and injecting current
        state_monitor = StateMonitor(neuron, ["v"], record=True)
        spike_monitor = SpikeMonitor(neuron)
        # run the simulation
        run(simulation_time)
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
        return TimedArray(np.transpose(big_matrix) * mA, dt = time)

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
                    dict_u_hat[values] = (((firing_threshold/(1-np.exp(-(d_u_hat/membrane_time_scale))))*(1/membrane_resistance)) /mA)*255
        else:
            for values in np.unique(spike_count):
                if values == 0:
                    dict_u_hat[values] = 0
                else : 
                    d_u_hat = simulation_time/values
                    dict_u_hat[values] = (((firing_threshold/(1-np.exp(-(d_u_hat/membrane_time_scale))))*(1/membrane_resistance)) /mA)

        temp = np.ndarray((shape[0],shape[1]))
        if is_pixel : 
            for i in range(len(spike_count)):
                temp[np.int64(np.floor(i/shape[0]))][i%shape[1]] = np.int64(np.floor(dict_u_hat[spike_count[i]]))
        else:
            for i in range(len(spike_count)):
                temp[np.int64(np.floor(i/shape[0]))][i%shape[1]] = dict_u_hat[spike_count[i]]

        return temp

    def fit(self, X, simulation_time=66 * ms, v_rest=0 * mV,
            v_reset=0 * mV, firing_threshold=0.09 * mV,
            membrane_time_scale=7 * ms, membrane_resistance=550 * mohm, abs_refractory_period=0 * ms,
            logger=False, is_pixel=True):
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
            BrianLogger.log_level_debug()

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

    def getEntropy(self):
        """
        Return the Entropy  of the signal, don't use getEntropy before fit
        """
        if not isinstance(self.spike, type(None)):
            return self._compute_entropy(self.spike.count)
        else:
            raise AttributeError("You cannot call getEntropy before fit")























# CUSTOMISABLE CODE START HERE

# a colored pic.
# Even if it's monochrome its pixel values will be grayed out anyway
myPic = "pics/hawk64.png"

# initial neuron value (probably in mV)
neuronVolt = 50; #0

# represents for how long (in MS) the neuron group is run. The longer the better the quality.
precision = 500; # 66
# the minimum mV for a neuron to fire
minFiring = 0.09 # 0.09

membrane_R = 550; # 550

# the refractory period in ms after a neuron spikes. During this time the neuron shouldn't be able to spike no matter what
abs_refract_time = 0; # 0

# the value in mV the neuron is set to after spiking. Usually they reset to the 0 mV value
neuron_reset_v = 0.0; # 0

# in seconds (?) - what does this even do?
memb_time_scale = 7; # 7


# print some brian-related debug stuff
brianDebug = True; # False










       

# plot the input (a 2 dimension array with [0,1] values)
def doAdisplay(inputPic, message):
    print(message + "\ninitial neuron v = " + str(neuronVolt) + " mV" +

          "\nduration = " + str(precision) + "ms\nfiring threshold = " + str(minFiring) + "mV\nmembrance Resistance = " +
          str(membrane_R) + "mohm\nNeuron refractory time : " + str(abs_refract_time) + "ms\nneuron reset value = " + str(neuron_reset_v) +
          " mV\nmembrane time scaling = " + str(memb_time_scale) + " s");
    plt.imshow(inputPic, cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0) # tweak vmin and vmax for some degradation
    #plt.plot(inputPic, 'cmap = gray')
    plt.show()


# initial neuron value (probably in mV)
# higher values feel like brightning up the neurons (pixels) and 0 means I guess default pixel color
neuronVolt = 0.0; #0

# represents for how long (in MS) the neuron group is run. The longer the better the quality.
precision = 66; # 66
# the minimum mV for a neuron to fire
minFiring = 0.3 # 0.09

membrane_R = 550; # 550

# the refractory period in ms after a neuron spikes. During this time the neuron shouldn't be able to spike no matter what
abs_refract_time = 0; # 0

# the value in mV the neuron is set to after spiking. Usually they reset to the 0 mV value
neuron_reset_v = 0.0; # 0

# in seconds (?) - what does this even do?
memb_time_scale = 7; # 7


# print some brian-related debug stuff
brianDebug = True; # False



rWeight = 0.2989;
gWeight = 0.5870;
bWeight = 0.1140;
        
def rgb2gray(rgb):
   return np.dot(rgb[...,:3], [rWeight, gWeight, bWeight])


# even if input is already grayed, it converts the 3D nparray into 2D array

# turns input picture into a 3D array representing rgb
myPicTreated = plt.imread(myPic); #Image.open(myPic).convert("L")
# gray it out, turning it to 2D
myGrayPic = rgb2gray(myPicTreated)









# LIFIFICATION


myLif = Lifq_2d()   


#doAdisplay(myGrayPic, "the pic which was just converted to grayscale")




def lifify():
    myLif.fit(myGrayPic, is_pixel=False,
                          simulation_time=precision * ms,
                          firing_threshold = minFiring * mV,
                          membrane_resistance = membrane_R * mohm,
                          abs_refractory_period = abs_refract_time * ms,
                          v_reset = neuron_reset_v * mV,
                          membrane_time_scale=memb_time_scale * ms,
                          v_rest = neuronVolt * mV,
                              logger=brianDebug);
    myPicBackToLife = myLif.getDecodedSignal()

    doAdisplay(myPicBackToLife, myPic)


        






