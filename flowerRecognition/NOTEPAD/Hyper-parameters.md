
# Hyper-parameters
_Machine learning models have hyper-parameters that you must set in order to customise the model to your dataset._

_Often the general effects of hyper-parameters on a model are known, but how to best set a hyper-parameter and combinations of interacting hyper-parameters for a given dataset is challenging. There are often general heuristics or rules of thumb for configuring hyper-parameters._
<br>

## Model Hyper-parameter Optimisation
Machine learning models have hyper-parameters.

Hyper-parameters are points of choice or configuration that allow machine learning model to be customised for a specific task or dataset.

_Hyper-parameters are set of model configuration arguments specified by the creator to guide the learning process for a specific dataset. Models also have parameters too which are the internal coefficients set by training or optimising the model on a training dataset. But they are called **parameters,** **NOT** **hyper-parameters** since **hyper-parameters** are set manually unlike automatically set **parameters**._  

Typically a hyper-parameter has a known effect on a model in the general sense, but it is not clear how to best set a hyper-parameter for a given dataset. Further, many machine learning models have a range of hyper-parameters and they may interact in non-linear ways.

As such, it is often required to search for a set of hyper-parameters that result in the best performance of a model on a dataset. This is called **hyper-parameter optimisation**, **hyper-parameter tuning**, or **hyper-parameter search**.

An optimisation procedure involves defining a search space. This can be thought of geometrically as an n-dimensional volume, where each hyper-parameter represents a different dimension and the scale of the dimension are the values that the hyper-parameter may take on, such as real-valued, integer-valued, or categorical.

-   **Search Space**: Volume to be searched where each dimension represents a hyper-parameter and each point represents one model configuration.

A point in the search space is a vector with a specific value for each hyper-parameter value. The goal of the optimisation procedure is to find a vector that results in the best performance of the model after learning, such as maximum accuracy or minimum error.

A range of different optimisation algorithms may be used, although two of the simplest and most common methods are random search and grid search.

-   **Random Search**: Define a search space as a bounded domain of hyper-parameter values and randomly sample points in that domain.
-   **Grid Search**: Define a search space as a grid of hyper-parameter values and evaluate every position in the grid.

Grid search is great for spot-checking combinations that are known to perform well generally. Random search is great for discovery and getting hyper-parameter combinations that you would not have guessed intuitively, although it often requires more time to execute.

More advanced methods are sometimes used, such as **Bayesian Optimisation** and **Evolutionary Optimisation**.

One of the most efficient method is **Bayesian Optimisation** so let's go through inside of it.
<br>

### Bayesian Optimisation
Provides a principled technique based on Bayes Theorem to direct a search of global optimisation problem that is efficient and effective. It works by building a probabilistic model of the objective function called the **surrogate function** that is then searched efficiently with an acquisition function before candidate samples are chosen for evaluation on the real objective function.

Bayesian Optimisation is often used in applied machine learning to tune the hyper-parameters of given well-performing model on a validation dataset.

_Bayesian optimisation is a powerful strategy for finding the extrema of objective functions that are expensive to evaluate. […] It is particularly useful when these evaluations are costly, when one does not have access to derivatives, or when the problem at hand is non-convex._ 

Recall that Bayes Theorem is an approach for calculating the conditional probability of an event:

$P(A|B) = P(B|A) * P(A) / P(B)$

We can simplify this calculation by removing the normalising value of $P(B)$
and describe the conditional probability as a proportional quantity. This is useful as we are not interested in calculating a specific conditional probability but instead in optimising a quantity.

$P(A|B) = P(B|A) * P(A)$

The conditional probability that we are calculating is referred to generally as the posterior probability; the reverse conditional probability is sometimes referred to as the likelihood and the marginal probability is referred to as the prior probability; for example:

$posterior = likelihood * prior$

This provides a framework that can be used to quantify the beliefs about an unknown objective function given samples from the domain and their evaluation via the objective function.

We can devise specific samples ($x1, x2, …, xn$) and evaluate them using the objective function  $f(xi)$  that returns the cost or outcome for the sample $xi$. Samples and their outcome are collected sequentially and define our data  $D$, e.g.  $D = {xi, f(xi), … xn, f(xn)}$ and is used to define the prior. The likelihood function is defined as the probability of observing the data given the function  $P(D | f)$. This likelihood function will change as more observations are collected.

$P(f|D) = P(D|f) * P(f)$

The posterior represents everything we know about the objective function. It is an approximation of the objective function and can be used to estimate the cost of different candidate samples that we may want to evaluate.

In this way, the posterior probability is a surrogate objective function.
<br>


