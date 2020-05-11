from brian2 import *



print(20*volt)

print(1000*amp)

print(1e6*volt)

print(1000*namp)

#print(5*amp+10*volt) # error: adding volts and amp (DimensionMismatchError)




tau = 10*ms
eqs = '''
dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1
'''
# dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1     #NeuronGroup method : euler
# dv/dt = (1-v)/tau : 1 #NeuronGroup method : exact


start_scope()

# neurons amount
neuronCount = 5

# method='exact'
G = NeuronGroup(neuronCount, threshold='v>0.8', eqs, method='euler')
# StateMonitor is used to record neurons properties while the simulation runs
# Takes args : neuron group, the property, and the neuron to record (cherry-picking neurons because too much neurons eats that RAM)
# record=True seems to be a synonym of "record all neurons"
M = StateMonitor(G, 'v', record=True) # True

initialVolt = 0

# it's possible to give an initial value (this changes every neuron's initial Voltage)
G.v = initialVolt

for i in range (len(G)):
    G[i].v += random()*8

# MY PERSONNAL ATTEMPT at modifying a single particular neuron's initial voltage :
#G[3].v = 13;
# works! G.v sets the volt of every neuron, G[i].v sets the volt of neuron of index i
#G.v = initialVolt #resets


run(50*ms)
for i in range (len(G)):
    plot(M.t/ms, M.v[i], 'C' + str(i), label='neuron'+str(i))


# the curve Brian produced
plot(M.t/ms, M.v[0], 'C0', label='Brian (equal to neuron0)')
# "analytic curve", or expected-value-curve (for another formula so unrelated to the other curves)
plot(M.t/ms, 1-exp(-M.t/tau), 'C1--',label='Analytic')
# my custom stuff for messing around with brian's features


xlabel('Time (ms)')
ylabel('v')
legend();

print('Expected value of v = %s' % (1-exp(-100*ms/tau)))
