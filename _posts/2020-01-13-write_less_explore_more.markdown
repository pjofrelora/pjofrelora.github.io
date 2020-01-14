---
layout: post
title:      "Write Less, Explore More"
date:       2020-01-14 00:19:02 +0000
permalink:  write_less_explore_more
---


I often get caught up in writing code — I focus on the code rather than what the code produces. This isn’t inherently bad practice, but in exclusion it doesn’t achieve the goal of Exploratory Data Analysis. The goal of EDA is to build a mental model of the data. If you spend 90% of your time on the code and 10% of the time on data, then you’ve only been 10% efficient at achieving your desired outcome. Knowing this, you should maximize the time that you spend exploring your data. Consider changing your process if you find that you’re spending a lot of time on EDA but don’t feel like you understand the data. One quick way to change your process is to ask yourself, “Is what I’m currently writing going to improve my understanding of the data, or is it helping me to achieve another goal?”

Realigning your process is a shortcut for better results, but changing your habits will have a more profound effect. Changing two habits has made an enormous impact on my exploratory data analysis: using simple graphics, and automating my process.

### Simple graphics communicate understanding

![If you squint and focus really hard you can see The Matrix! Oh wait, not, it’s just a regular matrix.](https://miro.medium.com/max/908/0*6fI2wvXMvxBr1iUA)

This is not a simple graphic. It contains simple elements, but as a whole it’s a complex array of information. Study it for 15 seconds. What do you see?

If your answer at the end of 15 seconds is, “Some features are correlated and some are not” then you’re in good company. You might go so far as to say, “there are some correlations, and some features are discrete while some are continuous.” Even better! As a whole though, you haven’t really learned anything, have you? You should already have known that some features were discrete when you imported the data, and I imagine you expected some things to be correlated. So all in all, you’re no better off after having looked at this graphic than you were before you looked at it!

This graphic is dense in information, but surprisingly light in meaning. The graphic doesn’t help you to develop understanding, and it doesn’t communicate your own understanding of the data. Remember that your goal is to develop a mental model — you do this by processing data into a series of relationships, not by memorizing information blindly . To do this efficiently, you should aim to produce graphics that strike a balance between information density and meaning density. That might be easier said than done. I’ll show you two examples to edify the concept and, hopefully, help you to improve your practice.

### Exploring Scatter Plots of Features with Discrete Data

Scatter plots of features with discrete data are a mess to look at. At best, you get a sense of the some of the descriptive statistics (mean, mode, min, max) of the feature and a vague sense of correlation. Here’s a scatter plot of a house price vs number of bedrooms in the King County Housing Data.

![What am I supposed to get out of this?](https://miro.medium.com/max/637/0*duMEHssNhVD5dVT-)

It’s just a bigger version of one of the graphs that you see in the scatter matrix above. In this case, we can’t even get a sense of how price responds to bedrooms because there are underlying relationships that we can’t see. In short, I can’t derive a lot of meaning from this graph because there’s too much information to process, and some information is hidden.

> Aim to Produce Graphics that Strike a Balance Between Information Density and Meaning Density

One way to determine the relationship between a discrete predictor and the target (if any exists) is to look at comparable data at each discrete value in the predictor. In order to do this we assume that the data behaves the same at each discrete value. That is, the distribution of the target at a discrete value (e.g. prices of houses with 3 bedrooms) has the same general shape as the distribution of the target for the other discrete values (e.g. prices of houses with 1, 2, 4, etc bedrooms). This allows us to compare the mean, median, or mode of the target at a given discrete value to evaluate for a relationship. It’s a bit tricky to think about, but the following graph should help crystalize the idea:

![](https://miro.medium.com/max/744/0*Yqe7OoIHvakJQKUi)

It looks like the relationship between price and bedrooms is linear, though there’s something going on with houses that have many bedrooms. A quick glance at the unique values in the predictor reveals that there simply aren’t that many houses with more than 8 bedrooms. We should only consider the data for houses with fewer than 8 bedrooms since this falls in line with our assumption about the behavior of the distributions.

### Deciding when (and how) to Bin

Deciding when and how to bin isn’t always so straightforward. Sometimes discrete groupings in features can be spotted in scatter plots, but usually you end up looking at a scatter plot that doesn’t make much sense. Here’s a not-particularly-useful scatter plot between Price and Lot Difference (Lot Difference is the scaled difference in lot size between a house and its 15 closest neighbors).

![](https://miro.medium.com/max/637/0*5pIDCGfe4dw91iL4)

It certainly doesn’t look like there is a relationship between Price and Lot Difference, but perhaps subsets of the data binned by quantile are predictive. The question is, how do you decide where to cut off your bins? In order to answer this you need to balance the information density with the meaning density of your graphic.
The purpose of binning data is to create groups that will have significantly different means from each other (µ1 ≠ µ2 ≠ µ3 … ≠ µn). In this case, the best graph overlays the distributions of the binned data so that you can quickly assess whether or not binning is appropriate.

![](https://miro.medium.com/max/730/0*H2keTL4pquPCwMnB)

It’s clear from this graph that binning this data won’t make it predictive of price, even if it’s binned at extreme quantiles. I am confident in making this statement because there isn’t a significant difference in the distributions of the different groups, which is something I can tell just by looking at the graph. Looking at the most meaningful graphic allowed me to understand the nuance in the data, which in this case means that I won’t end up with a meaningless predictor in our model.

### Use Simple Graphics
The success of the previous examples lie in their simplicity. At a glance you, and anyone else, are able to understand the relationships that are being described. You should aim to simplify your graphics whenever possible, following the tenet of balancing information density to meaning density.

### Automate Your Process
You’re probably not alone if you feel annoyed by the contradictory message so far. I started this post by asking you to “write less” and “explore more”, but so far I’ve told you that you should write more and explore more.
The truth is that you can’t explore without some friction, which in our case is writing code. So yes, you do have to write a little bit more code in order to explore more. The key, however, is to increase the ratio of exploring to writing. One of the best ways to do this is to reuse code effectively to make graphs automatically.

To do this, I employ ipywidgets. Ipywidgets is a simple module that allows you to interact with the parameters of a function on-the-fly by embedding controls into your output. It’s easier to explain with an example, so here’s a chunk of code and the resulting output.

```
@interact
def violin_plot(x=feature_list, yscale = ['log', 'linear'], hue = hue_list):
  fig, ax = plt.subplots(1,1,figsize = (12,8))
  if (hue != 'None') & (hue != x):
    title = "Price vs. " + x + " sorted by " + hue
    hue = kc[hue]
  else:
    hue = None
    title = "Price vs. " + x
  sns.violinplot(x = kc[x],
                 y = kc.Price,
                 cut = 0,
                 scale = 'area',
                 inner = 'box',
                 hue = hue,
                 ax = ax
                 )
  plt.yscale(yscale)
  plt.title(title)
```

![](https://miro.medium.com/max/753/0*2quOJVTiY-jq7wlk)

What’s not immediately apparent is that this chunk of code allows me to look at 60 different combinations of violin plots for this dataset! It’s a dizzying quantity, but the ability to choose the plot improves the likelihood that I will understand what I’m looking at. The reason for this is simple: I’ve prepared myself to update my mental model, and I can browse through plots with purpose.
Now that I’ve shown you ipywidgets, I can come clean about the examples in the previous section. Each one of those graphics was generated automatically using ipywidgets. They were each part of a larger investigation: the first example examined discrete predictors and their relationships to the target; the second example examined data for neighbors.
If you’re interested to learn more about ipywidgets, then I recommend that you read Will Koehrsen’s Medium post titled “Interactive Controls in Jupyter Notebooks” linked here. He walks you through how to use ipywidgets step-by-step with loads of examples. The documentation on ipywidgets is also very well written for those that want to go straight to the source.
Armed with ipywidgets, you should now feel comfortable exploring more of your data through simple graphics. Good luck, and happy exploring!
