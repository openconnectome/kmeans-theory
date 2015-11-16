---
author:
- Ron Boger
title: |
    Proof that the Bayesian Information Criterion is “unimodal” for a
    Gaussian mixture model
...

In applications of clustering, we often seek to best fit a model to data, but do not have prior knowledge of the number of clusters. Currently, the technique to find the best number of clusters to use is
to test a model at all relevant cluster sizes; however, given the prevalence of enormous data sets this method can prove to be extremely inefficient and costly. In this proof, we present an insight on how to
find the optimal number of clusters to use when for a more general case of the k-means problem - fitting k parameters to a Gaussian mixture
model.

The Bayesian Information Criterion (BIC) is a model selection tool. BIC can be informally defined as $BIC = penalty - fit$. Logically as we increase the numbers of parameters to be estimated, our fit improves, but the penalty worsens. We seek to minimize the BIC with respect to the number of parameters we estimate.

We begin the proof by providing important definitions and lemmas to aid in the readers’ understanding.

**Definition 1.1**: The BIC (Bayesian Information Criterion) is formally
defined as following:

$BIC = -2ln(\hat{L}) + k(ln(n) - ln(2 \times \pi))$, where:

$\hat{L}$ denotes the maximized value of the likelihood function for a model $M$, that is, $\hat{L} = p(x|\hat{\theta}, M)$. $\hat{L} \in \mathbb{R}$\

$\theta$ denotes a vector of the parameters of the model, and $\hat{\theta}$ is the estimator for $\theta$. $\theta \in \mathbb{R}^2k + (\frac{(k+1)(k)}{2} - 1)$

$k$ denotes the number of parameters to be estimated,
$k \in \mathbb{Z}^{+}$

$x$ is the observed data. $x \in \mathbb{R}^{n}$

$n$ is the number of samples in $x$. $n \in \mathbb{R}$

**Definition 1.2**: We consider a less traditional definition for the term *unimodal*. A function $f(x)$ is considered to be unimodal if all local extrema of the function are absolute extrema of the function. More
formally, this is:

- For $f(x) \in \mathbb{R}$, if $f'(x = x^{*}) = 0$, $f''(x= x^{*}) <0$, then ${\operatornamewithlimits{argmax}}_{x} f(x) = x^*$
- Similarly, for
$f(x) \in \mathbb{R}$, if $f'(x = x^{*}) = 0$, $f''(x= x^{*}) > 0$, then
${\operatornamewithlimits{argmin}}_{x} f(x) = x^{*}$

**Definition 1.3**: A Gaussian mixture distribution can be written as
follows:

$p(x) = \sum_{i=1}^{k} \pi_i N(x | \mu_i, \sigma_i)$, where:
$0 \leq \pi_i \leq 1$, $\sum_{i=1}^k \pi_i$

$N(x | \mu_i, \sigma_i)$ is a normal distribution
($N(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{ - \frac{(x - \mu)^2}{2\sigma^2}}$) with mean $u_i$ and variance $\sigma_i$

We provide more background on Gaussian mixture models:

Suppose we introduce $z = \{z_1, ... z_k\} $, where $z_i$ is a Bernouli random variable such that

$z_i = 1$ with $P(z_i) = \pi_i$ and $z_i = 0$ otherwise. We use $z_i$ to “single out” the $i^{th}$ $N(x | \mu_i, \sigma_i)$. Because $\bm{z}$ is only equal to $1$ for one value in $[k]$, we can write $p(z) = \prod_{i=1}^k \pi_i^{z_i}$, $p(x | z_i = 1 ) = N(x | \mu_i, \sigma_i) = \prod_{i=1}^k N(x | \mu_i,\sigma_i)^{z_i}$ 

and clearly $p(x) = \sum_z p(z) p(x | z) = \sum_{i=1}^k \pi_i N(x | \mu_i, \sigma_i)$

Furthermore, let us define the *responsibility* of $z_i$ as $\gamma(z_i) = \gamma(z_i = 1 | k) = \frac{p(z_i = 1)p(x| z_i = 1)}{ \sum_{j=1}^k p(z_j = 1) p(x | z_j = 1) } = \frac{\pi_i N(x | \mu_i, \sigma_i)}{\sum_{j=1}^k \pi_j N(x | \mu_j, \sigma_j)}$
by Bayes’ theorem.

**Definition 1.4**: Given $\{\bm{x_1}, \dots, \bm{x_n}\} = {X}$, where ${X}$ is independently and identically distributed, we define the *likelihood function* $L$ of ${X}$, $L = p( X | \pi, \mu, k) = \prod_{j=1}^n \sum_{i=1}^k \pi_i N(x_j | \mu_i, \sigma_i)$, and $ln(L) = \sum_{j=1}^n ln (\sum \pi_i N(x_j | \mu_i, \sigma_i) )$. 

We can maximize $L$ with respect to different parameters to attain $\hat{L}$, the *maximum-likelihood estimation (MLE)*.

Note that there is no closed form solution for the MLE of of a GMM.

**Theorem 1.1**: $\hat{L}(k = c+1) \geq \hat{L}(k = c) \forall c\geq 0$,
where $k$ again is the number of parameters to be estimated.\
**Proof**: We first seek to prove that $GMM_k \subset GMM_{k+1}$, where
$GMM_i$ is the set of all Gaussian mixture models with $i$ Gaussians
mixed. (Perhaps this part should be a lemma) Note that:\
$GMM_1  = \{P_{\theta} : P_{\theta}(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{ - \frac{(x - \mu)^2}{2\sigma^2}} \}, \mu \in \mathbb{R}, \sigma \in \mathbb{R}^+$\
$GMM_2 = \{P_{\theta} : P_{\theta}(x) = \pi_1 \frac{1}{\sigma_1 \sqrt{2\pi}} e^{ - \frac{(x - \mu_1)^2}{2\sigma_1^2}} + \pi_2 \frac{1}{\sigma_2 \sqrt{2\pi}} e^{ - \frac{(x - \mu_2)^2}{2\sigma_2^2}} \}, \mu_1, \mu_2 \in \mathbb{R}, \sigma_1, \sigma_2 \in \mathbb{R}^+ , 0 \leq \pi_1, \pi_2 \leq 1, \pi_1 + \pi_2 = 1$\
and,
$GMM_k = \{P_{\theta} : P_{\theta}(x) = \sum_{i=1}^k \pi_i \frac{1}{\sigma_i \sqrt{2\pi}} e^{ - \frac{(x - \mu_i)^2}{2\sigma_i^2}}  \}, \mu_i \in \mathbb{R}, \sigma_i \in \mathbb{R}^+ , 0 \leq \pi_i \leq 1\forall i, \sum_{i=1}^k \pi_i = 1$\
Examining this more careful definition of the set of all k-Gaussian
Mixture Models, we see can “recover” $GMM_k$ from $GMM_{k+1}$by setting
$\pi_{k+1} = 0$, and $\pi_{i, GMM_{k+1}} = \pi_{i, GMM_k}$, so
$GMM_k \subset GMM_{k+1} \forall k \geq 0$. Clearly the reverse
direction is not true.

For any 2 events $A, B$ in a sample space $S$, if $A \subset B$,
$Pr(B) \geq Pr(A)$. For our Gaussian Mixture Model, we can use this fact
and our previous definition of the set of all $k$-Gaussian Mixture
Models
$P(GMM_{k+1} | x) =  \sum_{i=1}^{k+1} \pi_i N(x | \mu_i, \sigma_i) \geq P(GMM_{k}  =  \sum_{i=1}^k \pi_i N(x | \mu_i, \sigma_i)$.
Since $ln$ is a monotonically increasing function,
$ln(\sum_{i=1}^{k+1} \pi_i N(x | \mu_i, \sigma_i)) \geq  ln(\sum_{i=1}^k \pi_i N(x | \mu_i, \sigma_i))$,\
$ \sum_{j=1}^n ln (\sum_{i=1}^{k+1} \pi_i N(x_j | \mu_i, \sigma_i) )\geq  \sum_{j=1}^n ln (\sum_{i=1}^{k} \pi_i N(x_j | \mu_i, \sigma_i) )$
so $\hat{L}(k = c+1) \geq \hat{L}(k = c) \forall c\geq 1 $\
**Lemma 1.1:** Concavity is “additive” in both the continuous and
discrete sense. That is, if $f(x) \in \mathbb{R}$ is concave up and
$g(x) \in \mathbb{R}$ is not concave down (has $0$ concavity or is
concave up), then $f(x) + g(x)$ is concave up. The same is true
replacing the prior statement with replacing all instances of “concave
up” with “concave down”.

**Proof:** The continuous case is trivial, as concavity is determined by
$sign(f(x))$ and $\forall f(x), g(x) \in \mathbb{R}$,
$\frac{\partial^2}{\partial x^2} (f(x) + g(x)) = \frac{\partial^2}{\partial x^2}f(x) + \frac{\partial^2}{\partial x^2}g(x)$,
so if $\frac{\partial^2}{\partial x^2}f(x) \geq 0$,
$\frac{\partial^2}{\partial x^2} g(x) \geq 0$ then
$\frac{\partial^2}{\partial x^2}f(x) + \frac{\partial^2}{\partial x^2}g(x) \geq 0$
and thus $h(x) = f(x) + g(x)$ is concave up with respect to $x$. Note
that this holds without loss of generality for
$\frac{\partial^2}{\partial x^2}f(x) \leq 0$
$\frac{\partial^2}{\partial x^2}g(x) \leq 0$for showing
$h(x) = f(x) + g(x)$ is concave down with respect to $x$.

We first seek to define concavity in the discrete sense. Let us refer to
a function as *concave up* over $[a, b] \in \mathbb{R}$ for
$f(x) \in \mathbb{R}$,
$f(x = k+2) - f(x = k +1) \geq f(x = k +1) - f(x = k)$, and a function
is concave down if
$f(x = k+2) - f(x = k +1) \leq f(x = k +1) - f(x = k)$ for some $k $ in
$[a, b-2] \in \mathbb{R}$.\
Let $f(x), g(x) \in \mathbb{R}$ for $x \in [a, b]$, and let us define
$h(x) = f(x) + g(x)$. Suppose
$f(x = k+2) - f(x = k +1) \geq f(x = k +1) - f(x = k)$ and
$g(x = k+2) - g(x = k +1) \geq g(x = k +1) - g(x = k)$. Then adding the
2 properties for $f, g$ together, we attain:\
$(f(x = k+2) + g(x = k+2))- (f(x = k +1) + g(x = k+1)) \geq (f(x = k +1) + g(x = k+1)) - (f(x = k) + g(x = k))$
, and thus\
$h(x = k+2) - h(x = k +1) \geq h(x = k +1) - h(x = k)$

Therefore $h(x)$ is concave up over $x \in [a, b]$ as well. Without loss
of generality, this applies to the addition of concave down functions
being concave down.

**Remark 1.1:** For a concave-up function $f(x) \in \mathbb{R}$, if
$f'(x) < 0 \forall x < c \in [a,b]$ and
$f'(x) > 0 \forall x > c \in [a,b]$, then $f(x)$ attains a local maximum
on the interval $[a,b]$ at $x = c$. This holds for our definition of
concavity in the discrete sense as well.

**Theorem 1.2:** It is sufficient to show that
$\hat{L}(k = c+2) - \hat{L}(k = c +1) \leq \hat{L}(k = c+1) - \hat{L}(k = c) \forall c \geq 1$
in order for the $BIC$ to be unimodal for a Gaussian Mixture Model. This
means the maximum likelihood function is in the discrete sense, concave
down.\
**Proof:** Let us recall our prior definition of the Bayesian
Information Criterion,
$BIC = -fit + penality = -2ln(\hat{L}) + k(ln(n) - ln(2 \times\pi))$.
Suppose that
$\hat{L}(k = c+2) - \hat{L}(k = c +1) \leq \hat{L}(k = c+1) - \hat{L}(k = c) < \infty \forall c \geq 0$.\
Then since our $penalty = k*(ln(n) - ln(2\pi))$increases linearly as a
function of $k$. we can find a $k^* < \infty$ s.t.
$k \times (ln(n) - ln(2\pi)) \geq \hat{L}(k=2) - \hat{L}(k=1) \geq \hat{L}(k=k^*) - \hat{L}(k=k^* - 1) \forall k > k^*\geq 1$.\
Therefore, since $ln(\hat{L})$ (and trivially our penalty) is a
monotonically increasing function as a function of $k$, $\exists k^*$
such that $BIC(k = c) - BIC(k = c-1) \leq 0 \forall 2 \leq c < k^{*}$
and $BIC(k = c) - BIC(k = c-1) \geq 0 \forall c \geq k^{*}$. Thus,
applying **Lemma 1.1** and **Remark 1.1** $BIC$ is unimodal for a
Gaussian Mixture Model as a function of $k$.

