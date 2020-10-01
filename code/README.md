# FloWPS user’s manual

FloWPS (FLOating Window Projective Separator) is an approach that improves the performance of P-vs-N classifiers. Such improvement is done via data trimming before the construction of machine leraning (ML) models. This data trimming is specific to any validation point.    

## Executable scripts

The deposited R package `flowpspkg.tar.gz` is designed for leave-out-out cross-validation of our approach. For any validation point, predictions of FloWPS classifier are calculated using `prediction.R`. 

The function 

flowps_main = function(
  train_file,
  out_dir=NULL,
  clf='LinearSVC', # [KernelRidge, AdaBoost, BernoulliNB, MLP, RandomForest, LinearSVC, SVC, knn]
  clf_args=list(),
  min_surround=0,
  max_surround=NULL,
  min_neighbours=32,
  max_neighbours=NULL
)

calculates prediction for every sample of the training dataset `train_file`. 

## Prerequisites 

Python 2 or 3: listed in `requirements.txt` and can be installed via standard `pip`:

```
pip install -r requirements.txt
```

R: requires `caTools` and `rjson` packages:

```
install.packages("caTools")

install.packages("rjson")

```

## Input data structure 

### CSV file:

* First column: sample ID
* Second column: observed classification scores for two (and only two) classes
* The rest columns: features

### Example

The exampels of datasets for this code may be found in the Supplementary Tables 2 and 3 for the paper (Tkachev et al., 2019; Front. Genet. 9:717. doi: 10.3389/fgene.2018.00717).

They are gene expression datasets with clinical response of individual cancer patients to certain anti-cancer treatment. In that case:

* First column: patient ID
* Second column: response which is 100 for clinical responders or 0 for non-responders
* The rest columns: various gene expressions

The script can be used for the following ML methods:

* Linear SVM (default method): LinearSVC, C is the user-defined cost/penalty parameter with default value C = 1
* Ridge (Tikhonov) regression: KernelRidge, with default parameter settings in the sklearn Python package
* Adaptive boosting: AdaBoost, with default parameter settings in the sklearn Python package
* Binomial naive Bayes: BernoulliNB, with the following parameter settings: "alpha":1.0, "binarize":0.0, "fit_prior":False; other parameters have default settings from the sklearn Python package
* Multi-layer perceptron: MLP, with the following parameter settings: "hidden_layer_sizes":(30, ), "alpha":0.001; other parameters have default settings from the sklearn Python package
* Random forest: RandomForest, with the following parameter settings: "n_estimators":30, "criterion":'entropy', "class_weight":'balanced_subsample'; other parameters have default settings from the sklearn Python package
* Polynomial SVM: SVC, C is the user-defined cost/penalty parameter with default value C = 1
* kNN: knn.

### Suggestions for *m* and *k* range selection and troubleshooting


The feature selection parameter *m* specifies the number of surrounding training points along the axis that corresponds to a feature, which is sufficient to keep this feature as relevant for building the ML model. Therefore, the value *m* = 0 corresponds to no feature selection. The theoretical maximum for *m* value is certainly *(N-1)*/2, where *N* is the number of samples in the dataset, since if *m* > *(N-1)*/2, then no feature would be chosen as relevant. In practice, there is a chance to encounter an error during the execution of the  script due to no relevant feature survived even when *m* < *(N-1)*/2, particularly for the datasets with relatively low number of features. In this case, please do not panic, and simply decrease the argument for maximal number of *m*, `max-surround`. 

The *k* parameter specifies the number of nearest neighbors in the subspace of selected features. Although this approach seems quite similar to the kNN machine learning method, unlike the kNN, where *k* is relatively small and does not usually exceed 20, we recommend to make *k* values for FloWPS higher. The maximal possible number for *k* is clearly *N*-1, which correspond to no training point selection, whereas at relatively low values there is a risk of another ML error, due to presence of only single-class training points among the *k* nearest neighbors. This trouble can also be easily fixed by increasing the  `min-neighbours` argument.

It is recommended to always keep the values of `min-surround`=0 and `max-neighbours` = *N*-1 since it never produces an error during the `flopws.py` execution, and provides a reference for comparison of FloWPS with classical ML methods without data trimming, which corresponds to *m* = 0, *k* =*N*-1. 

### Expected computation time

Please note that execution of code may be time-consuming due to wide ranges of *m* and *k* and a large number *N* of samples in the dataset. At a typical contemporary computer, e.g. with 31 Gb of RAM and 8*4.20 GHz CPUs, running the `flowps.py` script for our smallest cancer dataset , `TARGET20_with_busilfan`, which has 46 samples, takes about 20 seconds, whereas our biggest cancer dataset, `GSE25066`, which has 235 samples, will require a few hours to be calculated.  So, if the user intends to process a dataset, which is considerably larger than our cancer datasets, he/she should be warned about the computation time. Fortunately, the expected computation time can be easily assessed using the running log through the screen during the execution of `flopws.py`.  

## Output data 

Running this code produces the file `results/<dataset>/y_flowps_score.csv`.

Columns:
* `names`: sample ID
* `y`: observed classification score
* `flowps_score`: overall FloWPS predictions, *PF*. 

## Interpretation of results

The *PF* values are expressed in regression-like terms, i.e. they are likelihoods for attribution of samples to any of two classes. The discrimination threshold, which the user may apply to distinguish between two classes, should be determined according to the cost balance between false positive and false negative errors. In our cancer datasets example, we considered the costs for false positive and false negative errors to be equal, and then maximized the overall accuracy rate, ACC = (TP+TN)/(TP+TN+FP+FN), since the class sizes in our model datasets were equalized by design.

As far as it was mentioned before, it may be interesting to compare the performance of FloWPS to classical ML with no data trimming, so for this purpose one should run the script with *m* = 0, *k* = *N*-1.
