---
layout: post
title:  "Reverse Mode Automatic Differentiation"
date:   2021-09-05 00:00:00 +0100
categories: machine-learning
show_excerpts: True
excerpt: 'Example implementation of back propagation algorithm for training deep neural networks.'
featured_img: assets/auto-diff/model-output.png
---

---
<sup>__note__: *Relevant code for this post is [autograd-test](https://github.com/mauicv/autograd-test)*</sup>


## Intro

Exploration of a simple implementation of reverse-mode automatic differentiation. I start by giving a simple intro to the theory behind automatic differentiation and then introduce a very simple and inefficient example implementation. Finally I'll talk about generalizing this approach to neural networks and illustrate the results of doing so.


## Theory

There are tuns of articles on the theory and I won't dwell on it much as I'm mostly interested in the implementation. But Lets say  you have the following function $$f(x, y) = 2yx^2 + xy + 3$$. This function is made up of operations on two variables. Operations are just functions themselves. For instance the multiplication operation is a function. Call it $$m(x, y) = xy$$ and the addition operation $$a(x, y) = x + y$$. Using these we can write the above function $$f$$ as:

$$
f(x, y) = a(a(m(2, m(y, m(x,x))), m(x, y)), 3)
$$

Each of these operations is on two variables. So two values go in and one value comes out. This can be represented in a graph like so:
<div style="text-align: center">
  <img src="/assets/auto-diff/function-graph.png" style="text-align: center">
</div>

Each node in the graph is an operation. Information flows through the graph via the lines connecting the nodes. $$y$$ for instance flows as input into two multiplication operation nodes. This gives an intuitive understanding of the influence that $$y$$ has on the graph. A change in $$y$$ only has an effect on those nodes connected to $$y$$.

Note that we can consider a sort of localized picture of the graph using the variables $$a, b, c, d, e$$ that represent the outputs of each operation. For instance $$b = m(a, y)$$.

We might want to know what $$\partial f (x, y)/\partial y$$ is and the graph gives us an idea of how to do this. Following the flow of information from $$y$$ we can see which nodes we'll need to consider when computing the above. The nodes, in this case node, that we can ignore are those unconnected to $$y$$. Namely the $$m(x, x)$$ node as a change in $$y$$ here won't contribute to a change in that value and thus no change will propagate through to $$f$$. Using the chain rule we have:

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial b}\frac{\partial b}{\partial y} + \frac{\partial f}{\partial d}\frac{\partial d}{\partial y}
$$

In general starting at a node $$m$$ we can compute the deriviative using:

$$
\frac{\partial f}{\partial y} = \sum_{n}{\frac{\partial f}{\partial n}\frac{\partial n}{\partial m}}
$$

where $$n$$ is each node for which there is an edge in the graph flowing from $$m$$ to $$n$$. If we know $$\partial f/\partial n$$ and $$\partial n/\partial m$$ for each node $$n$$ connected to the node $$m$$ then we can reverse down the graph computing $$\partial f/\partial m$$.

The one issue is that at each node in order to compute $$\partial n/\partial m$$ we need to know the values of the inputs into $$n$$. We can do this by first running forward through the graph computing each of the inputs into each node.

## Implementation

Lets consider a simple implementation:

```py
class Edge:
    def __init__(self, from_node, to_node, dv):
        self.from_node = from_node
        self.to_node = to_node
        self.dv = dv

class Var:
    def __init__(self, value):
        self.in_edges = []
        self.out_edges = []
        self.df_dn = None
        self.value = value

    def new_node(self, other, value, dv_s, dv_o):
        node = Var(value)
        e1 = Edge(self, node, dv=dv_s)
        e2 = Edge(other, node, dv=dv_o)
        self.out_edges.append(e1)
        other.out_edges.append(e2)
        node.in_edges = [e1, e2]
        return node

    def compute_df_dn(self):
        if not self.out_edges:
            self.df_dn = 1
        else:
            self.df_dn = 0
            for edge in self.out_edges:
                self.df_dn += edge.to_node.df_dn * edge.dv
        return self.df_dn
```

The method `compute_df_dn` is equivalent to the derivative equation above. Given a node `n` the set `n.out_edges` is the set of connections to nodes dependent on `n`. let `e` be one such edge. Suppose at `m = e.to_node` we know `df_dn` which is how the final output changes with respect to that nodes output. Then we can compute the contribution to the total derivative given by changes that propigate along that path by multiplying `m.df_dn * e.dv`. Here `e.dv` is $$\partial m/\partial n$$ the change along the edge and `m.df_dn` is $$\partial f/\partial m$$ the change in `f` from the rest of the network flowing on from the connected node.

If we know the `e.dv` for all the edges in the graph then if we reverse through the nodes as they where created the `m.df_dn` should exist for each step. But we still need the `e.dv`. To get these values first notice that they're local in the sense that we only need to know the derivative of the node operation and the inputs into the node. Thus if we first run the graph forward computing the propagation of values and the derivatives then once we're done we'll have all these values for when we run the backwards pass.

A nice way to do this is to override the `__add__`, `__mul__` magic methods on the `Var` class so that when you say, add two instances of `Var`, you create a new node in the tree with the output as a value property and the correct derivatives on the edges.

```py
class Var:
    def __init__(self, value):
        ...

    def __add__(self, other):
        return self.new_node(other, value=self.value + other.value,
                             dv_s=1, dv_o=1)

    def __mul__(self, other):
        return self.new_node(other, value=self.value * other.value,
                             dv_s=other.value, dv_o=self.value)

```

so an operation on two instances of the `Var` class creates a new instance of `Var`. In order to ensure we can reverse through these nodes in the correct order we use a `Tape` class to record the creation of each node in the graph.

```py
class Tape:
    def __init__(self):
        self.nodes = []

    def watch(self, nodes):
        self.nodes.extend(nodes)
        for node in nodes:
            node.tape = self

    def compute_grads(self):
        for node in self.nodes[::-1]:
            node.compute_df_dn()
```

we'll need to change the `Var` class slightly as well so that it knows about the tape and records new nodes onto it.

```py

class Var:
    def __init__(self, value, tape=None):
        self.in_edges = []
        self.out_edges = []
        self.df_dn = None
        self.value = value
        # each Var has a reference to the tape
        if tape:
            self.tape = tape
            self.tape.nodes.append(self)

    def new_node(self, other, value, dv_s, dv_o):
        # Note when we create a new node we record it onto the tape.
        node = Var(value, tape = self.tape)
        e1 = Edge(self, node, dv=dv_s)
        e2 = Edge(other, node, dv=dv_o)
        self.out_edges.append(e1)
        other.out_edges.append(e2)
        node.in_edges = [e1, e2]
        return node
    ...
```

Thus we can now do:

```py
if __name__ == "__main__":
    x, y, c1, c2 = (Var(2), Var(3), Var(3), Var(2))
    tape = Tape()
    tape.watch([x, y, c1, c2])
    fn = c2 * y * x * x + x * y + c1
    tape.compute_grads()
    print('fn output value =', fn.value)
    print('x.df_dn \t=', x.df_dn)
    print('y.df_dn \t=', y.df_dn)
```

and this gives:

```
fn output value = 33
x.df_dn 	= 27
y.df_dn 	= 10
```

which from computing the derivative of $$f$$ w.r.t. $$x$$ and $$y$$ we see is correct. You can see the full implementation [here](https://github.com/mauicv/autograd-test/blob/main/src/reverse_mode.py).

## Back Propagation:

Reverse mode automatic differentiation is the general case of the back propagation method used for training neural networks. In a neural network for each layer you have the weights matrix, $$W^{(i)}$$, bias vector, $$b^{(i)}$$, and activation function $$\sigma$$. Given an input vector $$x^{(i)}$$ to the $$i^{th}$$ layer the output is given by:

$$
x^{(i+1)} = \sigma(W^{(i)}x^{(i)} + b^{(i)})
$$

Many such layers can be chained together to create a deep neural network, call it $$M$$.

If we have a training data set made up of $$x, y$$ pairs we want to choose the weights and biases such that $$l(M(x),y)$$ is minimized where $$l$$ is a loss function that compares the model output and the true output and returns a some measure of error between the two. If $$l$$ is differentiable then we can consider a function $$f(W, b) = l(M_{W, b}(x), y)$$. This is a function from the set of weights and biases to a model error. Namely from $$\mathbb{R}^{n} \rightarrow \mathbb{R}$$. We can build a graph made up of each operation in the sequence of layers similar to how we did for $$f(x, y) = 2yx^2 + xy + 3$$. This time however we're asking for the gradients w.r.t. the weights and biases instead of the $$x$$ inputs.

Using the above implementation would be pretty difficult and inefficient. Mostly because it requires functions be decomposed into only operations of two values. Lots of references are going to mean slow lookup times as operations and derivatives are computed.

The nice thing about neural networks however is that they're made up of lots of very similar operations. For instance matrix multiplication and vector addition. Fortunately the derivative rules over these operations are easy to derive and generalize for each element involved in the operation. This means a) we have a simple rule for applying these operations and b) we have a simple rule for computing the derivatives for each variable in the operations. Because there are general rules that can be mapped over the relevant variables, libraries like numpy that store arrays contiguously in memory can obtain significant speed ups. Similarly lots of the same operation can be parallelized over multiple processors.

Perhaps with a little bit of work you could extend the above implementation so that the value in the `Var` Class can be numpy arrays. You'd have to overide the `__matmul__` method for matrix multiplication too.

In the end I implemented a `Layer` and `Model` pattern [here](https://github.com/mauicv/autograd-test/blob/main/src/layer.py) and [here](https://github.com/mauicv/autograd-test/blob/main/src/model.py). In this case the Weight matrix and Biases are properties of the `Layer` class. Multiple layers are stacked in a `Model` and you can feed data through the model in the forward sweep. As you do so the local derivatives and values are computed. Then you can propagate the derivatives backwards to obtain the layer gradients.

I used the gradients to train a model that approximates the function $$f(x,y) = (x^2 + y^2)/2$$. The graph of the loss over the training duration looks like:
<div style="text-align: center">
  <img src="/assets/auto-diff/auto-grad-training-example.png" style="text-align: center">
</div>

And to compare the results the image bellow shows the true function on the left and the model output on the right:

![](/assets/auto-diff/model-true-compare.png)
