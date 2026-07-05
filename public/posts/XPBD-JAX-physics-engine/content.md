When I was first getting into software development I built a physics engine in javascript. You can see it [here](https://genesistubs.com/asteroids) - please be warned - it's not very good. It used an approach called particle based dynamics which is very nice. Anyway at the moment i'm building a robot and trying to train it to walk and as ever I get distracted by shiny ideas - one such shiny idea was could I model the robot using a differentiable PBD engine. This was attractive because in theory I could use the gradients through the engine to learn parameters that make the model more accurate. But also because I'm keen on [RL world models](#/posts/rl-world-model) - it's potentially possible to use a PBD engine as a prior which we then train a world model on top of. Anyway - I got some of the way doing this before I realised it's probably a much bigger job than i really have time to take on. Regardless - this blog post records some of what I learnt.

## XPBD

What is XPBD? XPBD stands for extended particle based physics. First lets quickly cover particle based physics (PBD). Particle based physics models systems using particles. Instead of an object with orientation and inertia you have a set of points that you tie together using constraints that cause the motions of each point to influence the others. As an example you could model a cube by a single center of mass with an orientation plus a measure of how mass is distributed in that cube. But then you have to deal with rotational mechanics etc... Instead in particle based dynamics you have 8 points - one for each corner and then you tie those points together with distance constraints that force those points to stay a certain distance from each other. All of this is pretty simple to do and you completely side step rotational mechanics. Things like collisions become much simpler. This is what the above gif of the cube is.

first step of the recipe goes as follows: 

- $$\circ$$ Each point is represented by a $$x_\text{current}$$ and a $$x_\text{previous}$$. 
- $$\circ$$ Each iteration compute $$v = x_\text{current} - x_\text{previous}$$
- $$\circ$$ Set $$x_\text{previous}$$ to $$x_\text{current}$$.
- $$\circ$$ Compute the new current x value using: $$x_\text{current} \rightarrow x_\text{current} + v$$. 

This is the inertial step and just says things in motion stay in motion. The next step of the process involves ensuring any constraints you require the particles to satisfy are enforced. An example of such a constraint is requiring particles to stay a distance $$d$$ apart, or $$C(x, y) = d(x_\text{current}, y_\text{current}) = d$$. Having performed the inertial step, some, likely all of the particles will no longer satisfy their constraints and so the next part of the process is to update their positions so that they do so. In PBD the way this works is you compute the gradient of the constraint function $$\nabla_x C(x, y) = - \nabla_y C(x, y)$$. This is a vector that points from each particle $$x$$ or $$y$$ to the other. You then just move each particle along their representative vectors until $$C(x,y)=d$$. so you're literally just pushing them back together. The process is actually a little more involved, you typically also take into account the mass but in this case you just update each position relative to how heavy each particle is. For the purposes of this just assume the particles are both the same weight. So you're moving each half of the distance along $$\nabla_x C(x, y)$$ or  $$\nabla_y C(x, y)$$ in order to ensure the constraint is satisfied.

There are multiple such constraints you can enforce. As an example the floor in the above gif is modelled as a plane at height $$z_\text{floor}=-1$$ and each particle satisfies a collision constraint $$C(\underline{x}) = \max(0, z_\text{floor}-z)$$ where $$\underline{x} = (x,y,z)$$. Thus the gradient for this constraint looks like an update pushing the particle out of the floor and back into the region $$z>z_\text{floor}$$. Typically you'll also want to add friction to contact constraints - in the case of the floor constraint this just looks like setting the $$x_\text{current}=x_\text{previous}$$ in the $$x$$ and $$y$$ axis. Doing so prevents the particle sliding along the floor. From all of this you can model rigid structures and things like angular momentum and inertia are sort of modelled implicitly by the distribution of particles and their constraints.

But how does XPBD differ. Essentially, PBD enforces hard constraints, so we try to snap points back into states that satisfy the constraint. XPBD instead models constraints as spring like and in particular allows you to model a stiffness. This helps for convergence but also makes it easy to model soft body dynamics. To understand how this works lets start by a very simple case. lets focus on a single particle constrained to the origin. Under PBD the constraint update would just teleport the particle straight back to the origin - so it wouldn’t be very interesting. In our case however we’re going to use XPBD which models the constraints as a soft “spring” update instead. Thus the constraint becomes a potential and movement from $0$ results in a restoring force towards the origin described by this potential. The potential is given by $U(x)=\frac{k}{2}\cdot x^2$. We can compute the force from this using $F = -\nabla U(x) = -k\cdot x$ which is the spring equation. 

The update process is still decomposed into two parts the inertial update:

$$
x_{\text{new}} → x_{\text{new}} + (x_{\text{new}} - x_{\text{old}})
$$

and the constraint update. Where in this case instead of working out how to project the points back onto the constraint manifold you instead have their movement governed by the potential given above.

### Side note on implicit and explicit dynamics.

When we discretize a differential equation to solve it using numerical methods we have a choice to make: suppose you have a differential equation: $\frac{dx}{dt}=g(x)$. This can be discretized by using the definition of gradient to get:

$$
\frac{x_{i+1} - x_{i}}{\Delta t} = g(x^*)
$$

note we’ve left $x^*$ to be chosen in $[x_i, x_{i+1}]$. As the above is an approximation to the gradient no choice in the range will be accurate - accuracy is only obtained by shrinking the size of $\Delta t$ - that is to say there’s not a “right” choice here.

However there is a simplest choice for the purpose of numerically solving these systems, namely choosing $x^*=x_i$ gives us:

$$
x_{i+1}=g(x_i)\Delta t + x_i
$$

The above is an equation that gives the next step in terms of the current step. Making this choice allows us to simply step forward the dynamics. This is known as an explicit solution to the differential equation.

Another choice however is $x^*=x_{i+1}$, which results in:

$$
x_{i+1}-g(x_{i+1})\Delta t - x_i = 0 
$$

Sometimes the above is solvable, but usually because there are two terms for $x_{i+1}$ we instead get an equation in $x_{i+1}$ that needs to be solved. Often to do this we use Newton's method. This is known as an implicit method. The key thing to understand about implicit methods is that they're more stable in stiff systems. Why? In the explicit method and a stiff system a particle that's jumped into a wall experiences a large force pushing it out of the wall. This is computed at the particle's current state within the wall (the explicit $$g(x_i)$$) this will tend to result in overshoot and a significant increase in energy in the system. In comparison the implicit method finds a $$x_{i+1}$$ that explains the movement the particle has experienced by the force applied at that point. This point must be inside the wall as the particle experiences no force outside the wall. As a result the implicit system undershoots the update and in fact these types of system are dissipative and lose energy over time.  

---

Back to the XPBD example, the explicit approach here would compute the force at $x_{\text{new}}$ and add its contribution but doing so tends to be unstable when working with very stiff constraints. Instead what we do is we perform the inertial update to get $x_{\text{new}}$ and then ask how to add force cumulatively (via a $\lambda$ parameter) to $x_{\text{new}}$  so that the sum of force and the displacement of $x$ satisfies $\lambda =-k\cdot x$. Another way of thinking about this is that we want to find the zeros of $g(\lambda) = x(\lambda) + \frac{1}{k}\lambda$. where note that $x(\lambda)$  is dependent on $\lambda$ because the force changes the position of $x$.

To do this imagine we’re updating x and $\lambda$ at the same time. The initial conditions are $(x=x_0,\lambda=0)$. If we plug those values in we get $g(\lambda)=x_0$ which obviously isn’t zero. We can use Newton's method to converge to the zero. This uses a sequence of estimates given by:

$$
\lambda \rightarrow \lambda − \frac{g(\lambda)}{g′(\lambda)}
$$

Hence we’ll compute $\Delta\lambda=-\frac{g(\lambda)}{g′(\lambda)}$ to then update $(x, \lambda)\rightarrow(x + \frac{1}{m}\Delta\lambda, \lambda+\Delta\lambda)$ and repeat for some number of steps. All that remains is to compute $\Delta\lambda$. To do this we first compute:

$$
g'(\lambda)=\frac{dx(\lambda)}{d\lambda} + \frac{1}{k}
$$

where $x$ is a function of $\lambda$ in that $x\rightarrow x + \frac{1}{m}\Delta\lambda$ which means $\frac{dx(\lambda)}{d\lambda}=\frac{1}{m}$. we’re going to simplify everything by assuming $m=1$ (we also are assuming the update rate, $dt=1$ as well) thus the derivative with respect to lambda is just 1. We get (we write $\alpha=\frac{1}{k}$):

$$
\Delta\lambda= -\frac{\alpha\lambda + x}{1+\alpha}
$$

And this is the update rule which you repeatedly apply to $(x, \lambda)$. Because we've simplified everything (like setting the mass to 1 etc) the $$\Delta\lambda$$ update is also the $$x$$ update. It's simple to code this above case up:

```python
import numpy as np
import matplotlib.pyplot as plt

k = 0.6
alpha = 1.0 / k

def solver(x, num_steps=4):
    lam = 0.0
    for _ in range(num_steps):
        C = x
        dlam = (-C - alpha * lam) / (1 + alpha)
        x = x + dlam
        lam += dlam
    return x

steps = 100
x_new, x_old = 10.0, 10.0
xs = []
for _ in range(steps):
    xs.append(x_new)
    x_predicted = x_new + (x_new - x_old)  # inertial update
    x_old = x_new
    x_new = solver(x_predicted, num_steps=4) # soft constraint update/s

plt.plot(xs)
```

this generates:

![particle-in-a-well](/posts/XPBD-JAX-physics-engine/xpbd-particle-in-a-well.png)

__Note__ that we never added any damping to this system and yet the particle has eventually fallen into the origin. This is the energy dissipation inherent in implicit systems. Although it's worth noting it's especially pronounced here as i've set the time step to 1 which is very high.

__Note__ If we set $$\alpha=0$$, which corresponds to x being connected to the origin by an infinitly stiff spring then we get: $$\Delta x= -x$$ which is exactly the PBD update. i.e. just move the particle back to the point where it satsifies the constraint.


### XPDB (more involved)

The above should have established an intuition for whats going on. Lets dive into the Paper a little deeper. We start with Newton's equations of motion:

$$
\begin{align}
M\ddot{x} = - \nabla U^T(x)
\end{align}
$$

We can discretize the above by turning $\ddot{x}={v_{i+1}-v_i}/{\Delta t}={(x_{i+1} - x_{i}) - (x_i - x_{i-1})}/{\Delta t^2}$ which gives us $\ddot{x}=(x_{i+1} - 2x_{i} + x_{i-1})/{\Delta t^2}$. If we choose the implicit choice for this discrete differential equation we get:

$$
M\frac{(x_{i+1} - 2x_{i} + x_{i-1})}{\Delta t^2} = - \nabla U^T(x_{i+1})
$$

The energy potential is just the spring constraints we mentioned in the intuition section. 

$$
U(x)=\frac{1}{2}C(x)^T\alpha^{-1}C(x)
$$

Where $$C$$ is your measure of constraint violation, in the particle in a well example above it's just the distance from the origin, in a length constraint between two particles it's the distance between them minus the default distance. 

The force in the above is given by:

$$
f_{\text{elastic}}=-\nabla_x U^T = -\nabla_x C^T\alpha^{-1} C
$$

If we introduce a variable $\lambda=-\tilde{\alpha}^{-1}C(x)$ then we can decouple the equations of motion (1) into two parts:

$$
\begin{align}
\mathbf{M}(\mathbf{x}^{n+1} - \tilde{\mathbf{x}}) - \nabla \mathbf{C}(\mathbf{x}^{n+1})^T \boldsymbol{\lambda}^{n+1} &= \mathbf{0} \\
\mathbf{C}(\mathbf{x}^{n+1}) + \tilde{\alpha} \boldsymbol{\lambda}^{n+1} &= \mathbf{0}
\end{align}
$$

$\tilde{x} = 2x^n-x^{n-1}=x^n+\Delta tv^n$ (the inertial update step mentioned in the intro) and where we also defined $\tilde{\alpha}=\alpha/\Delta t^2$

this is a system of equations of the form:

$$
\begin{align}
\mathbf{g}(\mathbf{x}, \boldsymbol{\lambda}) &= \mathbf{0} \\
\mathbf{h}(\mathbf{x}, \boldsymbol{\lambda}) &= \mathbf{0}.
\end{align}
$$

If we linearise, $$g$$ and $$h$$ we get:

$$
\begin{aligned}
  g(x, \lambda) &= g(x_i, \lambda_i)+ \nabla_x g(x_i, \lambda_i) \Delta x + \nabla_\lambda g(x_i, \lambda_i)\Delta \lambda_i \\

  h(x, \lambda) &= h(x_i, \lambda_i)+ \nabla_x h(x_i, \lambda_i) \Delta x + \nabla_\lambda h(x_i, \lambda_i)\Delta \lambda_i
\end{aligned}
$$

and then set $$g$$ and $$h$$ equal to 0, we have:

$$
\begin{aligned}
  \nabla_x g(x_i, \lambda_i) \Delta x + \nabla_\lambda g(x_i, \lambda_i)\Delta \lambda_i &= -g(x_i, \lambda_i) \\

  \nabla_x h(x_i, \lambda_i) \Delta x + \nabla_\lambda h(x_i, \lambda_i)\Delta \lambda_i
&= -h(x_i, \lambda_i)\end{aligned}
$$

this is a Newton's method update, we’re basically following the gradient of $h$ and $g$ from $(x_i, \lambda_i)$ to 0. Computing those gradients and substituting into matrix form we get:

$$
\begin{bmatrix}
\mathbf{K} & -\nabla \mathbf{C}^T(\mathbf{x}_i) \\
\nabla \mathbf{C}(\mathbf{x}_i) & \tilde{\alpha}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x} \\
\Delta \boldsymbol{\lambda}
\end{bmatrix}
=
-\begin{bmatrix}
\mathbf{g}(\mathbf{x}_i, \boldsymbol{\lambda}_i) \\
\mathbf{h}(\mathbf{x}_i, \boldsymbol{\lambda}_i)
\end{bmatrix}
$$

where $K=\frac{\partial g(x, \lambda)}{\partial x}$. The above system can be solved for $\Delta x$ and $\Delta \lambda$ and those values updated:

$$
\begin{align}
\boldsymbol{\lambda}_{i+1} &= \boldsymbol{\lambda}_i + \Delta \boldsymbol{\lambda} \\
\mathbf{x}_{i+1} &= \mathbf{x}_i + \Delta \mathbf{x}.
\end{align}
$$

Doing so repeatedly is just Newton's method.

The authors make two simplifications of the above by assuming two things. Firstly they look at the $K$ term which is:

$$
K=\frac{\partial g(x, \lambda)}{\partial x} = M - \nabla^2_x C^T(x)\lambda
$$

they simplify this by dropping the second term. Because $\lambda=-\frac{\Delta t^2}{\alpha}C(x)$, this only introduces $\mathcal{O}(\Delta t^2)$ error. 

Secondly they assume that $g(x_i, \lambda_i)=0$ for all $i$ which is justified with: 

> 
> This assumption is justified by noting that it is trivially true for the first Newton iteration when initialized with $x_0 = \tilde{x}$ and $\lambda_0=0$. 
> In addition, if the constraint gradients change slowly then it will remain small, and will go to zero when they are constant.
> 

Thus we get:

$$
\begin{bmatrix}
\mathbf{M} & -\nabla \mathbf{C}^T(\mathbf{x}_i) \\
\nabla \mathbf{C}(\mathbf{x}_i) & \tilde{\alpha}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x} \\
\Delta \boldsymbol{\lambda}
\end{bmatrix}
=
-\begin{bmatrix}
\mathbf{0} \\
\mathbf{h}(\mathbf{x}_i, \boldsymbol{\lambda}_i)
\end{bmatrix}
$$

We can kick these around for $\Delta \lambda$ to get:

$$
\left[\nabla \mathbf{C}(\mathbf{x}_i) \mathbf{M}^{-1} \nabla \mathbf{C}(\mathbf{x}_i)^T + \tilde{\alpha}\right] \Delta \boldsymbol{\lambda} = -\mathbf{C}(\mathbf{x}_i) - \tilde{\alpha} \boldsymbol{\lambda}_i.
$$

and for $\Delta x$:

$$
\Delta \mathbf{x} = \mathbf{M}^{-1} \nabla \mathbf{C}(\mathbf{x}_i)^T \Delta \boldsymbol{\lambda}.
$$

The above are linear equations for all points and constraints in the system. Instead of solving this directly they use a Gauss-Seidel Update in which they iterate
through the constraints and solve each in place one by one. Doing so means they can write each constraint update explicitly as:

$$
\Delta \lambda_j = \frac{-C_j(\mathbf{x}_i) - \tilde{\alpha}_j \lambda_{ij}}{\nabla C_j \mathbf{M}^{-1} \nabla C_j^T + \tilde{\alpha}_j}.
$$

the Gauss-Seidel step is an approximate solve, not exact, but it converges over repeated iterations of the process.

## Building a differential PBD physics engine using JAX

JAX is a python library that has a couple of key features - first it allows you to optimize functions using kernel fusing and secondly it allows you to compute gradients of functions very easily using autograd. Kernel fusing is a way of optimizing code to use less costly memory read/write operations and in so doing significantly increases the performance of code. In particular when you have a python function that does something like:

```python
x = 0
y = x + 2
z = y - 1
```

the dynamic compilation will convert this into something like

```
write 0 -> x
read x
add x, 2
write y
read y
add y, -1
write z
```

Instead if we recognise that the read and writing of intermediate values is unnecessary we can turn the above into:

```
write 0 -> x
read x
add x, 2
add x, -1
write z
```

On the other hand autograd is a method of efficiently computing gradients over trees of differentiable operations. See this [autograd post](#/posts/auto-diff) for more details of the idea. 

Autograd in particular allows us a way of abstracting a lot of the code we'd have to write in such a physics engine. Instead of computing the constraint gradients explicitly we can do it using JAX. Note that in the update equations for $$\Delta \mathbf{x}$$ and $$\Delta \lambda_j$$ the one complex term is the gradient of $$C$$ which needs to be computed analytically or... computationally using JAX. I built a 3D XPBD engine here called [Tac](https://github.com/mauicv/tac/tree/main), the following illustrates the `AutogradConstraint` class which allows the user to define a constraint by passing in the constraint function, the JAX autograd functionality computes the gradient in place and uses this in the update step. Technically doing so is slightly less efficient than precomputing the analytic gradient and just coding that in - however all of this is just math ops which are really fast, and with JAX's just in time compilation we can get rid of any unnecessary memory read/writes - resulting in pretty fast performance.  

One thing worth being clear about though - the autograd here is just computing the *constraint* gradient $$\nabla C$$ inside the solver, it's not the same thing as differentiating a whole rollout end to end. That end to end version - backpropagating through the full simulation to learn parameters - is the thing i was actually excited about back in the intro, and it's a fair bit harder than just getting $$\nabla C$$ for free.


```python
from tac.core.state import State
from flax import struct
import jax.numpy as jnp
import jax
from typing import Callable


@struct.dataclass
class AutogradConstraint():
    idx: jnp.ndarray
    alpha: jnp.ndarray
    rest: jnp.ndarray

    lambda_shape: tuple[int, int]
    C: Callable = struct.field(pytree_node=False)
    grad_C: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            points: jnp.ndarray,
            alpha: jnp.ndarray,
            rest: jnp.ndarray,
            C: Callable,
        ):
        points = jnp.asarray(points, dtype=jnp.int32)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)
        rest = jnp.asarray(rest, dtype=jnp.float32)
        n, k = points.shape
        lambda_shape = (n, 1)
        grad_C = jax.grad(lambda x: C(x).sum())

        return cls(
            idx=points,
            alpha=alpha,
            rest=rest,
            lambda_shape=lambda_shape,
            grad_C=grad_C,
            C=C,
        )

    def apply(self, state: State, lam: jnp.ndarray, dt: float, **kwargs):
        _alpha = self.alpha[None, :, None] / dt**2
        x, w = state._get_x_w(self.idx)                                   # (B, n, k, 3), (B, n, k, 1)
        C = self.C(x) - self.rest                                         # (B, n)
        grad = self.grad_C(x)                                             # (B, n, k, 3)
        g2 = (grad * grad).sum(-1)[..., None]                             # (B, n, k, 1)
        w_sum = (w * g2).sum(-2)                                          # (B, n, 1)
        residual = -(C[..., None] + _alpha * lam)
        denom = jnp.clip(w_sum + _alpha, min=1e-6)
        d_lambda = residual / denom                                       # (B, n, 1)
        upd = x + w * grad * d_lambda[:, :, :, None]                      # (B, n, k, 3)
        state = state.update_x(self.idx, upd)
        return state, lam + d_lambda
```

__Note__: The above blocks sets of constraints together so that instead of computing one at a time we do so for an entire block of them. This is more performant - however you have to be careful. Gauss-Seidel updates are sequential and depend on the previous update to particle positions in order to ensure stability. If you do them all at the same time particles can overshoot their constraint targets. As a result, we partition the set of constraints so that each only touches a point once, doing so is a compromise between efficiency (computing all update at once) and stability (prevents updates overshooting).


## Conclusion:

Anyway - that all gets you a pretty simple but impressive physics engine. 

![box falling](/posts/XPBD-JAX-physics-engine/box.gif)

I also added things like joint constraints which allow you to specify an angle between two points and an axis around which that angle is enforced - by allowing the rest value for these constraints to be set dynamically during rollouts we have a very simple model of a robot actuator. The following illustrates two separate "parts" connected via a joint which is driven using a simple sinusoidal signal. 

![actuated joint](/posts/XPBD-JAX-physics-engine/joint.gif)

As the aim of this project was a simulation of a robot i'm building and trying to train to walk I put together a simple proof of concept that models the robot with all its degrees of freedom as a wireframe. Here the joint positions are being randomly perturbed during the simulation - hence the jitter:

![actuated robot](/posts/XPBD-JAX-physics-engine/gnoci.gif)

I chose to leave this here for now because my instinct is there will be lots of unexpected details to solve in order to get it performant enough to actually apply to this robot i'm trying to train to walk. The differentiable side of things is also still an open problem for me - getting clean gradients back through the iterative Gauss-Seidel solve and the non-smooth contact updates is exactly the kind of thing that turns into a much bigger job than it first looks, and it's the part i'd actually need for the learning story i mentioned at the start. I do love this approach to physics though and I will return to it at some point. 