## Validation Tools
Development of statistical tests and tools for comparison of neural activity data with the aim of validating neural simulations.

Based on Elephant and neo.

#### ToDo

+ GED (generalized eigenvalue decompositon): de Cheveigné and Parra "Joint decorrelation ..."
    + can be calculated by scipy.eig

+ Make ValidationBook presentation-ready and integrate into collab
+ Separate Data (Results & Input) into non git controlled dir (to Sciebo)
+ Draw raw distributions for visual comparison
+ analytic T-dependent spectral radius
+ Apply Methods to PSTH Distribution Comparison
    + s. Paper von Grün, Abeles
+ Explore Parameters
    + Blaustein Cluster: Slurm queuing, rsync, requirements (-> Michael)
        + Performance estimate: 2.6MB/run, 11.2s/run (based on 32 runs)
        + Git checkout on cluster?
        + elephant, neo installation?
    + estimate minimum data size to detect features with a certain resolution
        + assemblysize; assemblycorr; assemblycorr with backgroundcorr  
+ How to evaluate the significance of high-dim angles?
    + Forward problem to mathematician!
+ Theoretical distribution of correlation matrix eigenvalues
+ Visualisation in -N/ K-space? -> DataHigh (for Win?)
+ PCA directly on binned spiketrains/ inst. firing rate (t)
    + Reduction of spiking variability by trial averaging/ temp. smoothing
    + Activity trace in <=3D k-space
    + luczak harris paper, rate vector, GPFA (-> Junji, Carlos)
+ Implement oscillations analysis (test with brunel)
    + spiketrainwise autocorrelation function
    + population autocorrelation
    + relation to fourier analysis
+ Write function annotations
+ Maybe apply method to Networks with ASSET method (->carlos)
+ Integrate CuBIC (in elephant)
+ Adapt code for collab packages

#### ORGANIZATIONAL CONSTRUCT

* METHODS

    + Distribution Comparisons
    + Correlation Matrix
    + Oscillations
    + (Graphs)

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
        + measures to detect state
    + SpiNNaker Izhekevich exact vs inexact