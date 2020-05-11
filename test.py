from brian2 import *



print(20*volt)

print(1000*amp)

print(1e6*volt)

print(1000*namp)

#print(5*amp+10*volt) # error: adding volts and amp (DimensionMismatchError)




tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''


start_scope()

G = NeuronGroup(1, eqs, method='exact')
# StateMonitor is used to record neurons properties while the simulation runs
# Takes args : neuron group, the property, and the neuron to record (cherry-picking neurons because too much neurons eats that RAM)
M = StateMonitor(G, 'v', record=0) # True

# it's possible to give an initial value
#G.v = 5

run(30*ms)

plot(M.t/ms, M.v[0], 'C0', label='Brian')
plot(M.t/ms, 1-exp(-M.t/tau), 'C1--',label='Analytic')
xlabel('Time (ms)')
ylabel('v')
legend();

print('Expected value of v = %s' % (1-exp(-100*ms/tau)))
