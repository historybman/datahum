---
Author: Bob Morrissey
title: class notes
---

# First day 
Different from digital humanities generally.

general mode:
- 1/2 lecture/discussion
- 1/2 hands on work

Normal office hour Tuesday 2-3pm.  For first three weeks Thursday 2-3pm too.  

## Ted's comments introduction to Data Science in the Humanities
- What is Data Science?
	- Relatively new phrase, but old concept
	- Drew Conway's [venn](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)
		- Danger Zone!  
	- David Donoho's article on 50 years of data science (have read this on the web)
		- Donoho argues that statisticians have already been doing this for a long time!
		- Rise of machine learning has produced a "predictive culture" in the last 25 years; producing a real philosophical change.  
		- Things that we could not model; things that we had to look at one by one, now we can model them.
	- What are the humanities? 
		- lots of overlaps but:
		- often care about the subtleties of cultural expression (art, music, literature)
		- humans care about _historical particularities_.  Not deducing general laws.  
			- this has been a barrier of combining data science with the humanities; it is a place where the humanities definitely has something to contribute to DS.
	-What can DS achieve in the humanities?
		- Ted's blog post : 7 ways humanists are using computers to understand text.
		- Low hanging fruit such as: what percentage of books written by women?
		- More sophisticated analysis- the emoji article; 
			- unsupervised, exploratory;
		- Scholars want evidence that can support arguments.
			- example: predictive model that can distinguish fiction from biography; the trendline shows that it _becomes easier_ to distinguish over time!  
		- Unstructured evidence is a lot like journalism or business
	- What can the Humanities bring to the DS?
		- Rosenberg, "Data Before the Fact"
			- how is data different from "fact"?  Raw.  "Things that are given."
		- Giteman, "Raw Data is an Oxymoron"-- basic premise: we HAVE the data because somebody selected it.  Somebody captured it.  The processes of selection always need to be considered.  

	- Statistics are a big challenge... Data sets are a problem.  


## Python Beginnings
### Data Science and Containers
- Data science is taking things and putting things in containers, and doing stuff with the stuff in the containers.  The containers are called variables.  

## Lists
- lists are mutable: you can change them without declaring a new variable and putting the new + old together in the new variable.  
- Strings are Immutable: you can change them only by declaring anew variable, or declaring the new + old and overwriting the old variable.  
- trying to set variable containing a list equal to that same list with an append method running on it, will set that variable to NULL, and delete it, because the list is immutable... 

## List comprehension
- use the technique to calculate... mean length of words in a list.  

# January 24, 2017

## Exploratory Data Analysis
- John Tukey's move to EDA: acknowledged that we should be doing some exploration, not just confirming statements, as in math.  
- Why not just do confirmatory hypothesis testing
	-Anscombe's quartet: summary statistics: mean, regression line, median; they all are the same regression line, despite such very different data sets.  So you need to look at other stuff besides hypothesis testing.  
	- Humanists usually DO start with exploration; we need to keep our eye on the other side of the equation--
		- you can do CDA on the same dataset, but you should ideally confirm on a different set of data.  Why?
		- There may be things that are true abotu it sort of accidentally.  There are LOTS of patterns in a dataset.  You need to test against other data to make sure that the pattern is not "weak pattern."  If you confirm on a different dataset, that is one way of confirming.  
		- Franco Moretti: Most people don't do confirmatory analysis on a different dataset. 

# January 31, 2017

- Dataframe contains multiple data types; matrix properly is just numbers.

Stats discussion: what does it mean that there is a real, unlikely to be random difference between the cost of the films that pass vs. the films that fail the bechdel?  

There might be confounding variables; for instance, movies made for men are more likely to be more expensive (special effects, etc.); these suggest possibility of sexism for sure, but more complicated than simply raw bias against women and stories about women.   
		

# February 7, 2017
What can we do with unstructured data?

It's analagous to what we we have been doing... but different.

Finding distinctive words: what exactly does it prove?  

Two approaches
- most distinctive words
- sentiment analysis

# February 14, 2017
Bayes and stuff;
Very solidly 20th century statistics so far.  WE started last week to talk about the distance between two documents or two datapoints in a "space."  That was late 20th century, and that concept of vector "space" became important with internet search.  Different measures of similarity in a geometrical space; but it was not really machine learning.

## Machine Learning
Pragmatic approach to supervised machine learning.  
When we train a machine learning program/ algorithm; come up with a "model of the difference between spam and not spam": this is also a classifier.  Spam filter.  

P ( something is science fiction | set of features)
"probability that something is science fiction given some set of features"

The test is to see if it can separate the categories; if the model starts to fail, or if it stops distinguishing clearly (.1 to .9 becomes lots of .45 and .55).

DIstinction between defining a specific conditions, vs. training a predictive model.  

How to do it responsibly, and how not to fool yourself.  
If you look at it as an engineering problem; I want to separate two things, and I want some code that will do it.
- but you can also look and see that there are theories: a formalized philosophy-- not just true and false but degrees of confidence.  
- Some process that relates
	- a vector, X {x1, x2, x3....}
	- TO an output, Y, or _y_, or g. [a number that quantifies the degree of difference or a category g, which is "spam/ not spam"]
- Our job is to infer parameters, Theta {B1, B2, B3...}
- That describes the relation between X and y/g.
- THen we can use that vector on new data to predict Y or g-hat.  

Why isn't this just statistics
Leo Breiman's "data modeling culture"

We used to be interested in model validation-- give me some parameters and I'll try to tell you what measures are causally related to whatever condition.

Now we are trying to do things that are somewhat more complex; not just try to come up with causal variables, we accept the fact that we don't have real causal factors-- there are LOTS of variables, and the key is that we're not trying to figure out the goodness of fit; here we're dealing with models are TOO good on the data that you trained your model on.  Measuring it by predictive accuracy rather than goodness of fit.  THAT'S WHY IT'S NOT JUST STATISTICS.  

Generative versus discrimitive models. 
- Naive Bayes is a generative algorithm
- Why *naive?*.  

It looks at variables in the data and tries to predict the process that would have produced that data.  Makes a theory about how spam is generated, how it is created.  It won't be permitted to fit the data exactly; you can overfit the data.  But what you want to do is to generate a model that can be applied to other data.  

Bayes' rule
## A great intuition.  
posterior probability can be UPDATED.  

Video of Corey Chivers explaining Bayesian updating.  
Naivete of variable independence will help us avoid overfitting; the assumption is that the variables are independent.  

You encounter a word, you think this tweet is trump, and you make your prediction.  You get the next word, and update ()

Multinomial vs. bernouli;

To avoid the problem of your probability numbers multiplying down to vanishingly small numbers, you can add the logs of these probabilities.  So you can build a Log column into your model.  

# February 21, 2017
A little on linear regression.

Brieman's data model culture vs. the machine learning culture;
Inferential stats, but not ML: Ordinary least squares linear regression
Minimizes the squares between the coordinates and the line that can describe them.  Like springs, whose force is literally proportional to the square of the stretch.  

## Problem: as the model gets more sophisticated, it overfits the data.
Dealing with text the danger of overfitting goes up dramatically: 

Breiman's "algorithmic modeling culture": decision trees, neural nets: they have means to make things simplified, to avoid overfitting the model.  "Deliberate fuzziness."

In Breiman's black box: We are predicting, rather than explaining. Systematic attempt to generalize, to force the machine to generalize, rather than to create a law, a specific set of rules, a precise detailed explanation.  

The logic of machine learning: learning to define a false precision with a complex problem: we are no longer going to judge things based on goodness-of-fit, you are going to test it on a out-of-sample dataset, a different dataset.  Also pragmatically, you can control how much fuzziness in your approach.  
Why prediction in humanities?  Because the prediction reveals-- 

### Supervised model: learns from labeled examples
### Unsupervised model: not labeled examples.  
Unsupervised: finds new patterns in the data; the advantage of the supervised approach is that you have something to anchor them-- just the amount of novelty you want.  

##Linear regression: relating one continuous variable to another. Producing a continuous variable. 
## Classifcation: produce a categorical variable
	- But Most classification algorithms can also be "calibrated" to produce a continuous probability-of-belonging-to class X.
## Clustering (unsupervised).

## Classification algorithms:
	- Naive Bayes
	- Decision trees
	- Random forests
	- k- Nearest Neighbors
	- Logistic regression (a classifier, despite the name)
	- support vector machines
	- neural nets

## Bias and variance tradeoffs; overfitting and underfitting
You want the model just complex enough to fit the data but not too much!  

Precision and recall: how many of the results that we said were positive were actually positive, vs. what proportion of the ones that we should have gotten did we actually get? 

# February 28, 2017

## Quantitative formalism article w/ Moretti and Jockers

Are they producing new knowledge?

What is unsupervised learning?  Real quick tour... 

Week 7: Unsupervised Learning
- Infers structure in unlabeled data

## K-means clustering: if you ever have to cluster data points on a two dimensional plane.

- Start by randomly selecting k data points as the centroids of clusters
- Until things stabilize:
	- assign every real data point to the nearest centroid
	- move each centroid to the "center of gravity" defined by the average of all points assigned to it.  

Pretty basic assumption that the clusters are "convex", aka round.

There are no guarantees because if you start at the wrong locations with k you can wind up with one of the centroids not finding a cluster; How do you choose a k?  

Also k means works best in low dimensional data where distance is meaningful; in text this is often not the case.  

Other methods

## Hierarchical clustering

- Builds from the bottom up.  Let's find the closest pair, then the closest "n" group, until you get to a logical place to stop, or the number of clusters that you feel is meaningful.  

- How to decide how many clusters is tricky.

- Dendrogram will visualize these cluster relationships.

### Hierarchical linkage criteria

Choose clusters by different logics, different rationales like lowest maximium distance between any pair. 

### DBSCAN: density based spatial clustering.  

## Curse of dimensionality, though
- the problem with high-dimensional data is not *just* technical or geometrical. The problem is, there could be multiple kinds of similarity in this space (subsets of dimensions).
- Compressing them all into a single generalized 'similarity' is not guaranteed to produce meaningful results.

## Solutions? 
- "Topic Modeling": acknowledges that we're dealing with "multiple kinds of similarity." But it's complex enough for a separate day.
- Alternatively, we could try to _reduce_ the number of dimensions, and then proceed with clustering..

## Dimensionality reduction
- Principal Component Analysis (just factoring the matrix). Just like Latent semantic analysis.
- some complex matrix algebra involved here.  

## PCA
- The factoring used in PCA (singular value decomposition) has other applications too
- Choices about scaling can make things volatile.  

## Manifold learning
- MDS (multidimensional scaling) tries to preserve original relative distances in a 2-dimensional projection
- T-SNE is a more complicated probabilistic solution to that problem.

## Grains of Salt
- Interpretation always starts from some assumption.
	- assumptions like "most frequent words matter"; or "all the words matter"
- People are attracted to unsupervised learning because it seems not to be importing external assumptions
- But that means that the assumptions are calling from inside the house.

# March 7, 2017
Going over homework from Week 6. 

REgularization forces you to have a simple boundary; prevents you from overfitting. 

Discussion of Pasanek, Sculley; Horton et. al on the 

## Ted's comments for the deay: A menagerie of learning strategies.
A tour of the options; a tour of the menagerie.

Learning algorithm: updtae boundary by pulling it toward misclassified examples.

HOw Logistic regression is different: Instead of a step function, the linear input is mapped to [0,1] with a logistic function.  In effect this means the boundary is blurry.

Also a regression algorithm optimizes the whole set. Prevents overfitting.

## Ensemble methods
- A combination of several different models can often perform better than any single model. 
- Why? This should be counter-intuitive; bias - variance tradeoff. The risk of overfitting should become a problem here. 
- The reason is that the errors are _uncorrelated_; this can reduce bias without increasing variance. 
- Bagging (bootstrap aggregating): generate a lot of models and make them "vote." A bootstrap sample is a 2/3 sample; with the possibility of sampling some of them more than once. 
- Boosting also is like this; here you focus on misclassified examples). 

## Random forests
- A bootstrap aggregate of decision trees, each working in a random subspace of the whole feature space.

## Support Vector Machines
- The separating hyperplane is often underspecified by an imperative to "minimize error." THis is still basically the state of the art for text classification. 

THe basic strategy os we are going to minimize the error.  

How do we pick the hyperplane that distinguishes: You want to maximize the margin:
- SVM adopt the more demanding goal of "maximizing the margin."

## The kernel trick
distorts the problem space with another dimension in order to separate

## For humanists, and in the prive sector, optimizing accuracy is rarely the only goal
- Complicated ensembles are tricky to implement
- Complex algorithms can be hard to explain. 

## Questions are rarely binary
- Multi-class prediction(directly, or via one-vs-all, or one-vs-one, strategies).

# April 25, 2017

Neural network processing of images-- 

Piper's "gaps" vs. Wallach's

## Piper

- Generality
- Explicitness: reproducible and collective rather than agonistic and opaque
- Self-reflexivity: models foreground the limits of knowledge
	- there will be places in the model where you acknowledge an assumption
	- explicit about the limits of your theory
- Relevance

## Wallach

- Privacy
- Fairness
- Question-driven rather than data-driven
- Not just predictive, but explanatory and exploratory

Acknowledging your situatedness is a strength that humanities bring to the table. 

## What are we gaining or losing from the neural network algorithms?

## Blei's article

The article covers the new hotness of 2013...

Build computer critique repeat...

THinking about quantitative model as a practical process-- 

- Some going off-road involved

- A bit of math
	- clusters and dots; different assignments to different clusters; hidden variables and actual datapoints. 
	- it's a predictive model, so we're assuming some hyperparameters and some hidden variables, and with those two unknowns, we have to estimate the probability of the data points that we have.  WE make a model of that: and we can then , as we have done in the past, take a NEW document, and then test how likely that new document could have been created by the model.  

The Triage question!!  




