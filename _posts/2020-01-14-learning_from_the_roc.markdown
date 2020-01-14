---
layout: post
title:      "Learning from the ROC"
date:       2020-01-14 22:13:01 +0000
permalink:  learning_from_the_roc
---


The American education world, it seems, is just now catching up to what Data Scientists have known for years. You learn from your mistakes.

You’re not alone if it sounds farfetched that our education system is just now figuring this out. It’s an age old adage, but somehow it skipped applied education theory. Don’t believe me? Then believe Janet Metcalfe, a professor of psychology at Columbia, who published “[Learning from Errors](https://www.annualreviews.org/doi/10.1146/annurev-psych-010416-044022)” in 2017 (just three years ago!).

![](https://cdn-images-1.medium.com/max/7000/1*q619Xrs4HVZQpscZ5AZw9w.jpeg)

In this seminal paper, Dr. Metcalfe points to a wealth of evidence demonstrating how American teachers consistently ignore mistakes, and instead teach students methods (read: formulas) to arrive at the right answer. In a not-so-surprising twist of events, it turns out that teaching a method performs far worse than learning from mistakes. This is corroborated by the strategies employed by Japanese teachers (who teach via error analysis) and the international exam scores comparing those countries. If only there had been some cross pollination between the two disciplines! It’s evident to us, Data Scientists, that learning from mistakes is the best way to learn a concept. In her paper, Dr. Metcalfe goes on to demonstrate exactly how learning from mistakes is better, and goes so far as to quantify the conditions under which learning is optimized.
> # The loss that you choose to minimize changes what your machine learns.

It is this final part — the nuance in optimizing learning — that I want to discuss in the context of Data Science. Can we learn from Education, just as Education should have done from Data Science?

## Teachers:Students :: Data Scientists:ML Algorithms

Do you remember those analogies from standardized exams? A is to B as C is to? Well, here’s one for you: Teachers are to Students, as Data Scientists are to Machine Learning Algorithms. As a quick aside, I am a teacher and this analogy isn’t meant to demean a teacher’s work (it’s hard stuff!).

The analogy is perfect for our context, especially if we (significantly) simplify the role of a teacher. Consider that a teacher’s goal is to help a student learn to make correct predictions. The predictions are so good, in fact, that we call them “answers”. The teacher employs a set of strategies to help the student learn as quickly as possible. One of those strategies can be pointing out mistakes. The student then takes those mistakes, and (hopefully) analyzes them to understand why their predictions were incorrect. They then update their mental model of the concept, and try again. For humans, this process is **efficient but not fast**. Students learn a new concept with just a few examples, but it takes some time to integrate the information.

![Is this a good or a bad example?](https://cdn-images-1.medium.com/max/2000/1*u0FFIbxNq4bfGlndqWaguw.jpeg)*Is this a good or a bad example?*

A data scientist does exactly the same thing. Our goal is to help our ML algorithms make correct predictions. If their predictions our good enough, then the algorithms graduate “to the wild”. Just like teachers, Data Scientists point out the algorithm’s mistakes, at which point the algorithm analyzes the mistakes and updates its digital model. For computers, this process is **fast but not efficient**.  Algorithms can update their models quickly, but it takes many examples to change the model.

## Efficient Learning

Why do students, and by extension humans, learn efficiently? My hunch is that we analyze error in a very nuanced manner, which leads us to understand *why *a prediction was incorrect. This leads to an optimal update of our mental model, which allows us to learn from far fewer mistakes. It has the added benefit that we can optimize for getting the correct answer under specific conditions. Let me explain:

Imagine you’re an early human that is foraging. You just saw your friend Hu pick a berry, eat it, and die (tragic story, sorry). Fortunately for you, this is a learning moment! You go investigate the berry and notice that it is purple mottled with yellow. Two days later you’re foraging again and find berries that are purple mottled with yellow-green. Do you eat it? Likely no! The cost of making a mistake is very high, so even though this berry doesn’t exactly match the pattern it’s not worth the risk.

The correct answer in this condition is “eat whatever doesn’t kill us”. So the berry may have been incorrectly labeled poisonous (call it False Positive), but at least you didn’t die. You could be said to be optimizing for recall at the expense of precision.

Think of that choice of words. You (early human version of you) is optimizing for recall. That sounds suspiciously like what machine learning algorithms do. In fact, it is **exactly** what ML sets out to do. It optimizes a system by minimizing loss. The loss that you (data scientist) choose to minimize changes what your machine learns.

## Choose Your Loss Function

If the loss that you choose to minimize changes what your machine learns, shouldn’t you choose a loss to maximize the metric you’re interested in? In a perfect world, yes, but unfortunately you have to contend with the restrictions of a loss function.

Generally, a loss function needs to be smooth, and ideally it’s also convex. If it’s not smooth, then you can’t calculate a gradient from which to update the model parameters. If it’s not convex, then you’re not guaranteed to arrive at the global minimum, or sometimes to achieve any optimization at all. A loss function also can’t be “one sided” — measuring only for precision or recall, for example. A “one sided” loss function will push the model towards “all or none” behavior in order to minimize loss. As a final caveat, the loss function needs to be of the form:

![](https://cdn-images-1.medium.com/max/2000/1*3Jl4EglrWnILvXJgaomq6A.gif)

Where *y* is the ground truth vector, and ŷ is the probabilistic prediction.

These restrictions leave you with fewer options, but by no means are you option-less. If the metric that you’re interested in improving is the Receiver Operating Characteristic (ROC), for example, then you should train for its integral. You should think carefully about the metric that you’re interested in improving, and when possible write a loss function to optimize for that metric.

## Learning From the ROC

Speaking of which, how **do** you learn from the receiver operating characteristic? Luckily for us, a team figured out how to do this in 2003! [You can read their publication linked here.](https://pdfs.semanticscholar.org/df27/dde10589455d290eeee6d0ae6ceeb83d0c6b.pdf) This loss is not implemented natively in Keras, but fortunately for you it’s easy to implement your own loss function. The following code is a (very) slightly modified version from the now defunct TFLearn library.

```
def auROC(gamma = 0.03, p = 2):
  """ ROC AUC Score.
      Approximates the Area Under Curve score, using approximation based on
      the Wilcoxon-Mann-Whitney U statistic.
      Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
      Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
      Measures overall performance for a full range of threshold levels.
      Arguments:
          gamma:  float. 0 < gamma < 1. This is the separation distance that you aim to achieve between classes.
                                        Samples that are separated by at least this much won't contribute to the loss.
          p:      float. 1 < p. This modulates amount of error as the distance to gamma is achieved.
      """
    def auroc(y_true, y_pred):
        with tf.name_scope("ROCAUC"):

            pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
            neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

            pos = tf.expand_dims(pos, 0)
            neg = tf.expand_dims(neg, 1)

            # original paper suggests performance is robust to exact parameter choice
            gamma_value = gamma
            p_value     = p

            difference = tf.zeros_like(pos * neg) + pos - neg - gamma_value

            masked = tf.boolean_mask(difference, difference < 0.0)

            return tf.reduce_sum(tf.pow(-masked, p_value))
    return auroc
```


In order to train for the auROC, initialize this function in a separate cell. Then, reference this function when you compile the model:

```
model.compile(optimizer = ‘adam’, loss = auROC(), metrics = …)
```


