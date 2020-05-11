from brian2 import *



##print(20*volt)

##print(1000*amp)

##print(1e6*volt)

##print(1000*namp)

#print(5*amp+10*volt) # error: adding volts and amp (DimensionMismatchError)




tau = 10*ms
eqs = '''
dv/dt = (2+v)/tau : 1
'''
# dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1     #NeuronGroup method : euler
# dv/dt = (1-v)/tau : 1 #NeuronGroup method : exact


start_scope()

# neurons amount
neuronCount = 4

# starting volts for neurons and volt reset upon spiking
initialVolt = 2 + random()*2

print('It was decided that neurons will start and reset upon spiking at the random', initialVolt, ' Volts value')

# method='exact'
G = NeuronGroup(neuronCount, eqs, threshold='v>10', reset='v=' + str(initialVolt), method='exact')

# SpikeMonitor registers the times at which neurons spike
spikeMonitor = SpikeMonitor(G)
# StateMonitor is used to record neurons properties while the simulation runs
# Takes args : neuron group, the property, and the neuron to record (cherry-picking neurons because too much neurons eats that RAM)
# record=True seems to be a synonym of "record all neurons"
M = StateMonitor(G, 'v', record=True) # True


# it's possible to give an initial value (this changes every neuron's initial Voltage)
G.v = initialVolt


# MY PERSONNAL ATTEMPT at modifying a single particular neuron's initial voltage :
#G[3].v = 13;
# works! G.v sets the volt of every neuron, G[i].v sets the volt of neuron of index i
#G.v = initialVolt #resets

# give a random increase to each neuron's starting volt
for i in range (len(G)):
    G[i].v += random()*9



run(20*ms)

print("Times t at which a neuron spiked ===> ", spikeMonitor.t[:])
for i in range (len(G)):
    plot(M.t/ms, M.v[i], 'C' + str(i), label='neuron'+str(i))
    


# the curve Brian produced
plot(M.t/ms, M.v[0], 'C0', label='Brian (equal to neuron0)')
# "analytic curve", or expected-value-curve (for another formula so unrelated to the other curves)
plot(M.t/ms, 1-exp(-M.t/tau), 'C1--',label='Analytic')
# my custom stuff for messing around with brian's features


# let's add a visual line on the graph on a neuron spike
for t in spikeMonitor.t:
    axvline(t/ms, ls='--', c='C5', lw=2)


xlabel('Time (ms)')
ylabel('v')
legend();

#print('Expected value of v = %s' % (1-exp(-100*ms/tau)))
