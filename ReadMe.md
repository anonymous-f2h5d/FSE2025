
# Supplemental Materials

This repository contains the replication package for the paper "An Empirical Study on Release-Wise Refactoring Patterns".


## Introduction



We have organized the replication package into two folders and five Python files:

1. data: This folder contains all the data required to run the experiments.
2. appendix: This folder includes additional information about the paper, such as the complete list of code smells and metrics in the study, as well as detailed results regarding the quality changes associated with pattern switches.
3. images: includes the plots and figures of our study.
4. clustering.py: this class is responsible for clustering data and analyzing the features of each cluster.
5. patterns.py: this class is responsible for studying pattern evolution and calculating transition probabilities.
6. main.py: This file acts as the main script, calling different classes to generate our research question results.

Our code is based on the following packages and versions:
- numpy: 1.26.4
- matplotlib: 3.8.2
- scipy: 1.12.0
- tslearn: 0.6.3
- seaborn: 0.13.2
- h5py: 3.10.0
- statsmodels: 0.14.1
- scikit-learn: 1.4.0
- scikit-posthocs: 0.9.0
- rpy2: 3.5.16

The following code can be used to install all packages in the environment.
```bash
  pip install -r requirements.txt
```

As r2py allows Python programs to interface with R, make sure to install R in your system:
https://www.r-project.org/

To load the dataset unzip the data.zip file in the root directory of the project. You can use the command below:
```bash
  unzip data.zip
```

We recommend using Python version Python 3.10.12, R version 4.4.1 and every Python requirement should be met.

    
## Usage/Examples

We have the following code and functions available in main.py, which should run in the order of the RQs, as RQ2 and RQ3 require the clustering results from RQ1:
```javascript
from patterns import Patterns
from releaseQuality import ReleaseQuality
from clustering import Clustering

clustering = Clustering()
releaseQuality = ReleaseQuality()
patterns = Patterns()

### RQ 1 RESULTS ##
"""
    The cluster_and_analyze() function includes subfunctions for pattern mining and analyzing release-wise refactoring patterns.
"""
clustering.cluster_and_analyze()

### RQ 2 RESULTS ###
"""
    Calculates the relationship of each release-wise refactoring pattern with  quality metrics.
"""
releaseQuality.pattens_quality()

### RQ 3 RESULTS ###
"""
    Calculates transition probabilities, distributions among different stages, and their relationship with quality metrics.
"""
patterns.print_transition_matrix()
patterns.distributions_in_different_stages()
releaseQuality.switches_quality()






