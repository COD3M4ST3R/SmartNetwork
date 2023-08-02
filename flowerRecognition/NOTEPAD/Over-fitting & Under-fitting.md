
# Over-fitting & Under-fitting

_Supervised machine learning is best understood as approximating a target function($f$) that maps input variables($x$) to an input variable($y$)._

$y = f(x)$

_This characterisation describes the range of classification and prediction problems and the machine algorithms that can be used to address them._

_An important algorithm in learning the target function from the training data is how well the model generalises to new data. Generalisation is important because the data we collect is only a sample, it is incomplete and noisy._
<br>

## Generalisation in Machine Learning
In machine learning we describe the learning of the target function from training data as inductive learning.

Induction refers to learning general concepts from specific examples which is exactly the problem that supervised machine learning algorithms problems aim to solve. This is different from deduction that is the other way around and seeks to learn specific concepts from general rules.

Generalisation refers to how well the concepts learned by a machine learning model apply to specific examples not seen by the model when it was learning.

The goal of a good machine learning model is to generalise well from the training data to any data from the problem domain. This allows us to make predictions in the future on data the model has never seen.

There is terminology used in machines learning when we talk about how well a machine learning model learns and generalises to new data, namely over-fitting and under-fitting. 

Over-fitting and under-fitting are the two biggest causes for poor performance of machine learning algorithms.
<br>

## Statistical Fit

In statistics, a fit refers to how well you approximate a target function.

This is good terminology to use in machine learning, because supervised machine learning algorithms seek to approximate the unknown underlying mapping function for the output variables given the input variables.

Statistics often describe the goodness of fit which refers to measures used to estimate how well the approximation of the function matches the target function.

Some of these methods are useful in machine learning (e.g. calculating the residual errors), but some of these techniques assume we know the form of the target function we are approximating, which is not the case in machine learning.

If we knew the form of the target function, we would use it directly to make predictions, rather than trying to learn an approximation from samples of noisy training data.
<br>

## Over-fitting in Machine Learning
Over-fitting refers to a model that models the training data too well.

_When something looks too good to be true, it usually is._

Over-fitting happens when a model learns the noises in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impact the models ability to generalise.

Over-fitting is more likely with non-parametric and non-linear models that have more flexibility when learning a target function. As such, many non-parametric machine learning algorithms also include parameters or techniques to limit constrain how much detail the model learns.

For example, decision trees are non-parametric machine learning algorithms that is very flexible and is subject to over-fitting training data. This problem can be addressed by pruning a tree after it has learned in order to remove some of the detail that it has picked up.

_One of the first hint that model has over-fitting is analysing better accuracy performance on training dataset than validating dataset. In this case model has learned how to react on this specific training data instead of generalise patterns._
<br>

## Under-fitting in Machine Learning
Under-fitting refers to a model that can neither model the training data nor generalise to new data.

An under-fit machine learning model is not a suitable model and will be obvious as it will have poor performance on the training data.

Under-fitting is often not discussed as it is easy to detect given a good performance metric. The remedy is to move on and try alternate machine learning algorithms. Nevertheless, it does provide a good contrast to the problem of over-fitting.
<br>

## A Good Fit in Machine Learning
Ideally, you want to select a model at the sweet spot between under-fitting and over-fitting.

This is the goal, but is very difficult to do in practise.

To understand this goal, we can look at the performance of machine learning algorithm over time as it is learning a training data. We can plot both the skill on the training data and the skill on a test dataset we have held back from the training process.

Over time, as the algorithm learns, the error for the model the training data goes down and so does the error on the test dataset. If we train for too long, the performance on the training dataset may continue to decrease because the model is over-fitting and learning irrelevant detail and noise in the training dataset. At the same time the error for the test set starts to rise again as the model's ability to generalise decreases.

The sweet spot is the point just before the error on the test dataset starts to increase where the model has good skill on both the training dataset and the unseen test dataset.

You can perform this experiment with your favourite machine learning algorithms. This is often not useful technique in practice, because by choosing the stopping point for training using the skill on the test dataset it means that the test set is no longer “unseen” or a standalone objective measure. Some knowledge (a lot of useful knowledge) about that data has leaked into the training procedure.

There are two additional techniques you can use to help find the sweet spot in practice: resampling methods and a validation dataset.
<br>

## How to Limit Over-fitting
Both over-fitting and under-fitting can lead to poor model performance. But by far the most common problem in applied machine learning is over-fitting.

Over-fitting is such a problem because the evaluation of machine learning algorithms on training data is different from the evaluation we actually care the most about, namely how well the algorithm performs on unseen data.

There are two important techniques that you can use when evaluating machine learning algorithms to limit over-fitting:
* Use a resampling technique to estimate model accuracy.
* Hold back a validation dataset.

The most popular resampling technique is **k-fold** cross validation. It allows you to train and test your model k-times on different subsets of training data and build up an estimate of the performance of a machine learning model on unseen data.

A validation dataset is simply a subset of your training data that you hold back from your machine learning algorithms until the very end of your project. After you have selected and tuned your machine learning algorithms on your training dataset you can evaluate the learned models on the validation dataset to get a final objective idea of how the models might perform on unseen data.

Using cross validation is a gold standard in applied machine learning for estimating model accuracy on unseen data. If you have the data, using a validation dataset is also an excellent practice.
<br>
