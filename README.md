# GeneticAlg
This library is my personal introduction to evolutionary algorithms and typing in Python. The problem and most of the code are inspired by [this tutorial by Eric Stoltz.](https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35) You can read about parameter tuning [here](https://towardsdatascience.com/tuning-a-traveling-salesman-cadfd7d22e1c).

Note that is was done in Python 3.6, so the class annotation will not use `from __future__ import annotations`.

## Tests & Examples
Tests for this repository are in the file GA_tests.py. Uncomment the tests to see the output of the code at various points in the algorithm.

![](generations.jpeg)  ![](https://raw.githubusercontent.com/adinhobl/GeneticAlg/master/GA4TSM.gif)

(i.) The best individual's fitness from each generation.    (ii.) The route taken by the most fit individual for each generation.

## Future Things to Try
- Try different fitness functions other than inverse to see if algorithm converges faster/better, e.g. squaring, different integer
