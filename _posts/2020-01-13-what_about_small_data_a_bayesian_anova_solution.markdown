---
layout: post
title:      "What About Small Data? A Bayesian ANOVA Solution."
date:       2020-01-14 00:25:57 +0000
permalink:  what_about_small_data_a_bayesian_anova_solution
---


## What is small data?
As data scientists we’re trained to work with big data, and generally we’re very comfortable in that domain. Big datasets let us use any number of tools to prod and probe before we hone in on an answer. The data yields to almost every approach, though it has to be massaged every now and then to make things work. Notwithstanding diminishing returns, the more elbow grease we put in the more we get out.

What about small data, though? Does it fit neatly into the approaches that we already use? Think of small data as anything that your average Joe can wrap his head around without having to process it too much. For example, the ages of his coworkers and their current position in his company is a small dataset. Joe can quickly see the trend and spot an outlier in the dataset, provided that his cohort of workers is relatively small.

Take a look at this small dataset that shows the scores one judge awarded participants from different countries. According to our definition, it seems like we should to be able to draw an intuitive conclusion from it.

![](https://miro.medium.com/max/154/0*GuFyJD3RrKtNFGyq)

You’re right to think that USA has higher scores than Canada and Mexico, but is this a trend? Can you say with confidence that the USA is scoring higher than Canada and Mexico over the long term, or could this be a blip on the radar? Can Joe assert that older employees have a higher position in the company, or is that also a blip on the radar?

The trouble with small data is exactly that. It’s small. We can draw intuitive conclusions from it, but when it comes time to make decisions using the data we get stuck. This is particularly true if you’re a data scientists or an analyst whose job depends on making the right call from data. So what should you do? If your first thought is, “I’ll avoid small data”, then think again! Small data is ubiquitous, and sooner or later you’ll have to contend with using it. In fact, collecting a small dataset is sometimes the best route. It lets you action on the data faster, which makes you a more nimble data scientist.

## Making Sense of Small Data
Let’s go back to the dataset above to make some sense of it. Imagine that an olympic diving judge has just been suspended because he was caught giving higher scores to US competitors as compared to Canadian and Mexican competitors. He fessed up, took a retraining course, and signed an agreement banning him from the sport if his scores were found to be biased. The dataset shows the average scores that he awarded participants of different nationalities at four international competitions. You can assume that the other judge’s scores varied equally, but they may not match up with his scores. If you’re a data scientist working for the Olympic committee that oversees diving judges, what do you recommend?

![](https://miro.medium.com/max/1024/0*rzaB1wi2Vy7-KYuO)

This is quite a quandary, isn’t it? Your intuition tells you that the scores are biased, but you need proof. Your goal is to determine if the average scores between countries are the same (H0: µ0 = µ1 =µ2) or different (Ha: µ0 ≠ µ1 ≠ µ2), which means that you should run an Analysis of Variance, ANOVA, test to arrive at a conclusion. Let’s set α = 0.10 so that we have a low threshold to continue investigating. The results of ANOVA are show below.

![](https://miro.medium.com/max/431/0*N-PPQaUuM8y72jTu)

Even with our liberal confidence level we can’t reject the null hypothesis. Is it that our intuition is wrong? Not necessarily! Our intuition is based on the fact that we have previous knowledge about the behavior of this particular judge. In other words, we’re primed to look for bias because we have evidence that there has been bias in the past. The problem that we’re running into is one of small data. There simply isn’t enough data to reject our hypothesis, and we have no way to include our prior knowledge in this analysis. Waiting to get more data in this scenario harms competitors and risks lowering the viewer’s confidence in the ability to judge a contest fairly. Luckily, we’re not stuck.

## Bayesian to the Rescue!
Bayesian statistics are perfect for this problem. If you haven’t yet encountered bayesian statistics, then I suggest that you [read this article](https://towardsdatascience.com/bayes-theorem-the-holy-grail-of-data-science-55d93315defb) and then [read this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4158865/pdf/cdev0085-0842.pdf). Using prior information is one of the reasons why bayesian statistics is powerful. Analyzing a small dataset such as this one is possible only because we have prior information that we feed into the model. By coupling the prior information to the new information using Markov Chain Monte Carlo methods, we can infer the likelihood that the new information fits within our prior understanding. Let’s run the analysis again, but this time using a Bayesian framework.

## Bayesian ANOVA in Python
ANOVA is functionally equivalent to simple linear regression using categorical predictors. In fact, the F-statistic for ANOVA is exactly the same as the F-statistic in linear regression for the model that only uses categories as its predictors. To get a better sense of what is being computed, we can leverage the confidence interval for the coefficients in order to test the hypothesis at a certain confidence level. If the 95% confidence intervals overlap for every coefficient, then we expect the F-statistic to yield a p-value > 0.05. The coefficients of linear regression for the data shows this relationship.

![](https://miro.medium.com/max/507/0*SDF3mIVUPGAPeG7P)

We can use this information, that linear regression is equivalent to ANOVA, to run a bayesian analysis in Python using the PyMC3 module. My code is copied below.

```
with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = pm.HalfCauchy('sigma', beta=10)
    usa = pm.Normal('USA', mu = 80, sd=1.9)
    can = pm.Normal('Canada', mu = 78, sd=1.9)
    mex = pm.Normal('Mexico', mu = 76, sd=1.9)

    # Define likelihood
    likelihood = pm.Normal('Score', 
                           mu = usa*df['USA'] + can*df['Canada'] + mex*df['Mexico'],
                           sd=sigma, 
                           observed=df.Score)

    # Inference!
    trace = pm.sample(3000, cores=2) # draw 3000 posterior samples using NUTS sampling
```

What’s going on here? First I load up a PyMC3 model (pm.Model() as model). Then, I stuff the model with the prior information. Sigma is the standard deviation that we expect in the whole model. I set sigma to be drawn from a half Cauchy distribution (standard practice), which confines sigma to be greater than 0 with higher probability of being a low value than a high value. The parameter beta sets the width of the distribution — a value of 10 is a generous width with values from 0–10 being essentially equally likely.

I then load the model with the bias that the judge has demonstrated in the past. The average scores for US, Canadian, and Mexican competitors were very similar, but he scored US competitors 2 points higher on average than Canadian competitors, and 4 points higher on average than Mexican competitors. This small difference would be hard to spot using a frequentist approach.

Finally, I define the likely results as the sum of each coefficient multiplied by the dummy variable that encodes for a competitor’s nationality, and compare it to the actual data, df.Score. In order to obtain results, I run two markov chains, each with 3000 samples.

How do I determine if there is bias? I leverage the concept of looking at the confidence intervals of the coefficients in the frequentist method! Instead of looking at the confidence interval, I look at the probability distributions for each coefficient and determine if the 90% highest posterior densities (HPD) overlap. The following graph shows that the HPDs do not overlap between Mexico and USA, which is evidence that the average scores are not equal.

![](https://miro.medium.com/max/716/0*-bsMpWJH_Z-atzCV)

Now that we have evidence that there could be bias, we should continue to investigate to ensure that the scores do not actually reflect the performance of the competitors. We can do this by comparing the scores that the other judges assigned to competitors with those of this particular judge. Given the small amount of data, I can imagine that these analyses would also have to be performed with Bayesian statistics.
So there you have it! Bayesian statistics is another tool to manage data, specifically small data, to make better recommendations or draw conclusions. Take a look at the PyMC3 documentation for more information on how to use that powerful library. If you’re interested in the full code for the examples above, then take a look at [my notebook linked here.](https://github.com/pjofrelora/BlogNotebooks/blob/master/ANOVA%20in%20Bayesian%20Framework.ipynb)
