## ValidationTools
Development of statistical tests and tools for comparison of neural activity data with the aim of validating neural simulations.

Based on Elephant and neo.


#### ToDo

+ estimate minimum data size to detect features with a certain resolution
    + assemblysize; assemblycorr; assemblycorr with backgroundcorr
+ How to evaluate the signficiance of high-dim angles?
    + Foward problem to mathematician!
+ Visualisation in -N/ K-space? -> DataHigh (for Win?)
+ PCA directly on binned spiketrains/ inst. firing rate (t)
    + Reduction of spiking variability by trial averaging/ temp. smoothing
    + Activity trace in <=3D k-space
    + luczak harris paper, rate vector, GPFA
+ Implement oscillations analysis (test with brunel)
    + spiketrainwise autocorrelation function
    + population autocorrelation
    + relation to fourier analysis
+ Write function annotations
+ Maybe apply method to Networks with asset method (->carlos)
+ Implement CuBIC (in elephant)
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