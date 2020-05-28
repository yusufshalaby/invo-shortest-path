# Shortest Path Inverse Optimization
Code for the paper "An Inverse Optimization Approach to Measuring Pathway Concordance", submitted to Management Science.

The inverse shortest path problem recovers the arc costs of a graph using known shortest paths.
We extend this framework by allowing for supplementary 'good' and 'bad' pathways in our data. The output costs minimize the total costs of the good pathways while maximizing the total costs of the bad pathways. We also offer two metrics for evaluating the 'goodness' of out-of-sample pathways. 

The code was used in the paper to model the clinical care network for a cohort of colon cancer patients, where 
clinical guidelines are used to create shortest paths, and 
survived (died) patient pathways are used as supplementary 'good' ('bad') data points. 
Guideline concordance for out of sample patient pathways is measured using our two 'goodness' metrics. 

## Example
The example below creates a network with activities ranging from A through J and two options for how to start and end.

```python
from CareNetwork import *
from InverseModels import *

activities = ['A','B','C','D','E','F','G','H','I','J']
start_options = ['start option 2','start option 1']
end_options = ['end option 2','end option 1']

network = SingleStateModel(activities,start_options,end_options)
```
You can define a shortest path and visualize it on the graph.

```python
shortestpath = ['START','start option 1','A','C','E','G','H','end option 2','END']
fig = network.showNetwork(path=shortest_path)
```
![](https://github.com/yusufshalaby/invo-shortest-path/blob/master/imgs/example_graph.png)

The code below creates synthetic good and bad data points and solves the inverse model to get weights for the graph.

```python
import random
n_good = 50
n_bad = 50
goodpaths = [['START','start option 1'] + random.choices(steps[:5],k=10) + ['end option 2','END'] for i in range(n_good)]
badpaths = [['START','start option 2'] + random.choices(steps[5:],k=10) + ['end option 1','END'] for i in range(n_bad)]

network.getCoefficients([shortestpath],goodpaths,badpaths)

fig = network.showNetwork(show_weights='c',show_labels=False)
```
![](https://github.com/yusufshalaby/invo-shortest-path/blob/master/imgs/example_graph_weights.png)

Once you have weights you can measure pathway concordance.
Below we generate new random good and bad pathways 
and measure their concordance scores using the optimized weights. 

```python
goodpaths_outofsample = [['START','start option 1'] + random.choices(steps[:5],k=10) + ['end option 2','END'] for i in range(n_goodpoints)]
badpaths_outofsample = [['START','start option 2'] + random.choices(steps[5:],k=10) + ['end option 1','END'] for i in range(n_badpoints)]

omega_good = network.getOmega(vectorizePaths(network,goodpaths_outofsample))
omega_bad = network.getOmega(vectorizePaths(network,badpaths_outofsample))

sns.distplot(omega_good,label='Good paths')
sns.distplot(omega_bad,label='Bad paths')
plt.xlabel(r'Concordance score ($\omega(\hat{\mathbf{x}})$)')
plt.ylabel('Number of pathways')
plt.legend()
```
The good pathways are highly concordant while the bad pathways are highly discordant.

![](https://github.com/yusufshalaby/invo-shortest-path/blob/master/imgs/example_omega.png)

