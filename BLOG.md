# FLOW AND DIFFUSION MODELS
### Sampling and Generation Relevance:
For any object in general, there is no single correct representation, rather we have a set of all possible representations for a specific object/entity i.e. our data. This data distribution can also be looked at as a probability distribution, which is often represented by p_data which contains random variables (x_1,.....,x_n)

We make an assumption that, upon sampling a random variable x~p_data, we generate a image of the desired object

Hence we can summarise our generation task as the following:
Converting a distribution p_init into samples from p_data

## Flow Models: 

### Ordinary Differential Equations (ODEs)

A trajectory in flow models is denoted as X[0,k] → Rᵈ, where:
- X represents the state
- t ∈ [0,k] is the time parameter
- Rᵈ is the d-dimensional state space

The vector field is represented as u(x,t), often written as u_t(x), which describes the direction and magnitude of change at each point in space and time.

We follow the direction vectors in the vector field across time from a fixed initial point, leading to a trajectory defined by:

```

```

A flow is a collection of solutions for an ODE with multiple different initial conditions.

**Note**: We assume unique solutions to these ODEs exist.

### Numerical Methods

One common method for numerical ODE simulation is Euler's Method:

```
X_{t+Δt} = X_t + u_t(X_t) * Δt
```

More sophisticated methods like Runge-Kutta can also be used for better accuracy.

## Model Architecture

The core idea is to convert an initial distribution p_init into a data distribution p_data using an ODE where we represent the vector field as a Neural Network (u_t^θ).

The process works as follows:
1. Sample initial point: X_0 ~ p_init (Random Initialization)
2. Evolve the point using the ODE: dX_t = u_t^θ dt

## Probability Paths

### Conditional Probability Path

A conditional probability path is a path of change from p_init to p_data (i.e., from noise to data) conditioned on a target point.

We use the Dirac Delta Distribution (δ_z), where sampling from δ_z always returns z.

The conditional probability path is defined as:
- p_0(·|z): initial distribution conditioned on z
- p_1(·|z) = δ_z: final distribution is a point mass at z

This effectively converts a single data point into the distribution p_init.

### Marginal Probability Path

When z ~ p_data, we get a marginal probability path:
- z ~ p_data
- x ~ p_t(·|z) 
- Thus, x ~ p_t

The density of the marginal path is given by:

```

```

## Vector Fields

**Conditional Vector Field**:


**Marginal Vector Field**:
```

```

The marginal vector field is the flow training target for which we choose a suitable conditional vector field satisfying all conditions.

## Training Methodology

### Flow Matching

The aim is to make the neural network u_t^θ equal to u_t^target by finding suitable parameters θ.

Flow matching loss (Mean squared error) is defined as:

```

```

Where p_t is the marginal probability path.

### Conditional Flow Matching

The flow matching loss is not efficient to compute because finding the marginal vector field via integration is non-tractable. Therefore, we use conditional flow matching as the conditional vector field is tractable.

Conditional Flow Matching loss is represented by L_CFM (replace all marginal terms in L_FM with conditional terms).

By expanding the mean squared error term, we can prove that:

```
L_FM(θ) = L_CFM(θ) + c
```

Where c is a constant independent of θ.

Hence, minimizing L_CFM is equivalent to minimizing L_FM.

Diffusion models
The key difference between flow models and diffusion model is that SDE is used instead of ODE
### SDE(Stochastic Differential Equations):
X_t is a random variable for every 0 ≤ t ≤ 1
X : [0,1] → ℝ^d,  t ↦ X_t is a random trajectory for every draw of X
Note: Simulating this equation multiple times will lead to different trajectories as SDE's are designed to be random
These equations are constructed via Brownian Motion(A stochastic trajectory where X_0=0 and trajectory is continuous, increments have a Gaussian distribution with variance increasing linearly in time)



#DDPM 

In this process the We are adding noise step by step and the denoising it in the reverse process 
the formula for the forward process is written as where for some beta t , q is a form of a gaussian with given mean and variance as shown in the formula here using simple algebra all the terms are written in th form of alpha which kinda direly gives us a method to predict direct noise from Xo TO Xt


So why exactly are we using gaussian  property here  at least intuitively we can say that if we have a forward gaussian then we can have a backward gaussian at least approximately tbh. 
Now there is a mathematical known fact here that if a given  forward transition kernel the reverse transition can be approximately written as (given beta is very small its very important) where mew is a mean function for Xo and Xt

Now its safe to assume for a large T the final image after adding noise is approximately converted into a standard gaussian therefore q(Xt)=N(0,I)(standard gaussian )
Now using simple bayes rule we can say that 

Then by doing some simple replacements from the above derivation we can say that 

Now in order to find the P theta term in the given formula we can say

We just need to find the mew term from the formula,now in order to find mew lets go to the starting equation for Xt as we know in order to find mew we need two things Xo and Xt  and we can just rearrange this equation and then equation of Xo in terms of Xt and epsilon that means we need to find only epsilon in order to find mew function so the whole problem for mew gets reduced to this formula 




#REFERENCES
https://www.practical-diffusion.org/
https://youtu.be/B4oHJpEJBAA?si=rvtT_8Pa9ZjGGfcT
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/




