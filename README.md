## ValidationTools
Development of statistical tests and tools for comparison of neural activity data with the aim of validating neural simulations.

Based on Elephant and neo.


#### ToDo

+ remove autocorrelation in visualisation
+ estimate minimum data size to detect differences
+ background corr in assemblies
+ asset method (carlos)
+ How to evaluate the signficiance of high-dim angles?
+ luczak harris paper, rate vector, gpfa
+ calculate angles between subspace relative to chance distribution f(N)
+ Calculate corrcoef distribution
  and run test down to which mean corrcoef
  assemblies can be identified
+ Shuffle spiketrains before analysis and
  and try to resort them in reasonable groups
  (i.e) in groups whereto the largest eigenvectors point
  -> show sorted matrix as heatmap
+ Visualisation in -N/ K-space? -> DataHigh
+ PCA directly on binned spiketrains/ inst. firing rate (t)
    + Reduction of spiking variability by trial averaging/ temp. smoothing
    + Activity trace in <=3D k-space
+ Implement CuBIC (in elephant)
+ Implement oscillations analysis (fourier analysis)
+ Write function annotations
+ Create test case with Brunel network
    + Test methods for SR, SI, AR, AI
+ Run test validation for different network parameters
+ Develop useful to_file function
+ Adapt code for collab packages
+ Create correct requirements.txt



#### ORGANIZATIONAL CONSTRUCT

* METHODS

    + Distribution Comparisons
    + Correlation Matrix
    + Oscillations
    + Graphs

* IMPLEMENTATION

    + Validation Toolbox
        + dist
        + matrix
        + test_data
    + Jupyter Interface
    + => Collab
    + => Elephant

* WORKFLOW

    + Determine the focus of usage
        + Neuroscience
        + Deep Learning
        + Simulation
        + Experiment Data
        + ..?
        
    + Test against    
        + Neuron model
        + Spike Loss
        + Time steps
        + Numerical Integration
        + Network Model
        + Reference 
        (given by different architecture/hardware)
        + ...

    + Creating references
         + Brainscales models Comparison
         + Universal reference exact NEST?
         + Create reference by regarding distribution of test results

* USE CASES

    + NEST vs SpiNNaker
    + Brunel network with different solver/ time step/ precision
    + SpiNNaker Izhekevich exact vs inexact