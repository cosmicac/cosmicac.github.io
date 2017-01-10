---
layout: post
title:  "Some Intuition For Logistic Regression"
date:   2016-12-19
published: true
---

I have a terrible memory, so I'm going to write down some intuition about logistic regression to help me remember.

Say you're given $$n$$ data points $$x_i \in \mathbb{R}^d$$ for $$i \in 1 ... n $$, and each one belongs to either class $$y_0$$ or $$y_1$$. We want to be able to predict the class of new data point. We can approach this by trying to model the probability that a data point $$x$$ belongs to $$y_0$$. 

A natural model to consider is the following, where $$w$$ is a weight vector in $$\mathbb{R}^d$$, and $$p = P(x \in y_0)$$:

$$
w^Tx = p
$$

But we notice that both the range and domain of our function is $$(-\infty, \infty)$$, but a probability must be in $$[0,1]$$. $$w^Tx$$ will always have range $$(-\infty, \infty)$$, so what we are really looking for is a function $$f(p)$$ that has domain $$[0,1]$$ and range $$(\infty, \infty)$$. Then, we can try:

$$
w^Tx = \log{p}
$$

This is a step in the right direction, since $$log(p)$$ has a domain of $$(0, \infty)$$, and a range of $$(-\infty, \infty)$$. We can do better:

$$
w^Tx = \log{\frac{p}{1-p}}
$$

Perfect. Now $$\log{\frac{p}{1-p}}$$ has a domain of $$(0,1)$$ and a range of $$(-\infty, \infty)$$. We can now solve for $$p$$ to obtain a function, $$p(x)$$ that has range $$(0,1)$$ and domain $$(-\infty, \infty)$$. 

$$
\begin{align*}
w^Tx &= \log{\frac{p}{1-p}} \\
e^{w^Tx} &= \frac{p}{1-p} \\
e^{w^Tx} &= p + pe^{w^Tx} \\
p &= \frac{1}{1 + e^{-w^Tx}} \\
\end{align*}
$$

So, now we have a reasonable model, $$p(x)$$, for $$P(x \in y_0)$$ (notice that $$P(x \in y_1) = 1 - P(x \in y_0)$$, since we have only two classes). But our model depends on the weight vector $$w$$ - what's the optimal $$w$$? We can't directly calculate the truly optimal $$w$$, that is, the one that will perform the best on test data, but we can approximate it with our training data. Let's try maximum likelihood estimation to find the $$w$$ that maximizes the probability of our training data occuring given $$w$$. 

Let $$y_i = 1$$ if $$x_i \in y_0$$, and $$y_i = 0$$ if $$x_i \in y_1$$. Then, we can write our likelihood function as:

$$
\begin{align*}
L(w) = \prod_{i=1}^{n} p(x_i)^{y_i}(1 - p(x_i))^{1 - y_i}
\end{align*}
$$

Notice that each term of the product is either $$p(x_i)$$ or $$(1 - p(x_i))$$, depending on if $$y_i = 1$$ or $$y_i = 0$$. This is exactly the behavior what we want. We can take the log to get the log-likelihood. 

$$
\begin{align*}
log(L(w)) = \sum_{i=1}^{n}y_i\log{p(x_i)} + (1-y_i)\log{(1-p(x_i))}
\end{align*}
$$

Unfortunately, there isn't a closed form solution for $$w$$ that maximizes this function. But, we can formulate a cost function and run gradient descent or stochastic gradient descent. Maximizing the log-likelihood is equivalent to minimizing the negative, so we write our cost function $$J(w)$$ as follows. This is also known as the cross-entropy loss. 

$$
\begin{align*}
J(w) = -\sum_{i=1}^{n}y_i\log{p(x_i)} + (1-y_i)\log{(1-p(x_i))}
\end{align*}
$$

To formulate our gradient descent update, we need to find gradient of $$J(w)$$ with respect to $$w$$, $$\nabla_w J(w)$$. But before we do that, let's first find the derivative of $$p(x_i)$$ with respect to $$w$$, $$\frac{\partial p(x_i)}{\partial w}$$. It'll be useful in the next step. We have:

$$
\begin{align*}
\frac{dp(x_i)}{dw} &= \frac{\partial}{\partial w}(\frac{1}{1 + e^{-w^Tx_i}}) \\
				&= \frac{x_i e^{-w^Tx_i}}{(1+e^{-w^Tx_i})^2} \\
				&= (\frac{1}{1 + e^{-w^Tx_i}})(\frac{e^{-w^Tx_i}}{1 + e^{-w^Tx_i}})x_i \\
				&= p(x_i)(1 - p(x_i))x_i
\end{align*}
$$

Now, to compute $$ \nabla_w J(w) $$:

$$
\begin{align*}
\nabla_wJ(w) &= \nabla_w[-\sum_{i=1}^{n}y_i\log{p(x_i)} + (1-y_i)\log{(1-p(x_i))}] \\
		     &= -\sum_{i=1}^{n} \frac{y_i}{p(x_i)} \frac{\partial p(x_i)}{\partial w} - (\frac{1-y_i}{1-p(x_i)})\frac{\partial p(x_i)}{\partial w} \\
		     &= -\sum_{i=1}^{n} \frac{y_i(1-p(x_i)) - (1-y_i)p(x_i)}{(1-p(x_i)p(x_i)} p(x_i)(1-p(x_i))x_i \\
		     &= -\sum_{i=1}^{n}(y_i - p(x_i))x_i \\
		     &= -\bf{X}^T(\bf{y}-\bf{p})
\end{align*}
$$

In the last step, we converted our gradient into matrix notation, where $$ \bf{X} $$ is a $$ n \times d $$ matrix (each row is a data point $$ x_i $$), $$ y $$ is a vector where the $$i^{th}$$ element is $$y_i$$, and $$p$$ is a vector where the $$i^{th}$$ element is $$p(x_i)$$. Now we can write out our gradient descent equation, with $$ \alpha $$ as our learning rate:

$$
\begin{align*}
w_{i+1} &= w_{i} - \alpha\nabla_{w_i} J(w_i) \\
		&= w_{i} + \alpha\bf{X}^T(\bf{y}-\bf{p_i}) \\
\end{align*}
$$

As it turns out, our cost function $$ J(w) $$ is convex, so this should work very well.