When writing a code base with sufficiently complex class structures a problem that can arise is that when such complex class structures get themselves into an invalid state it can be many lines of code before that invalid state generates an error. This is frustrating as it can be enormously difficult to work backwards from an error in order to find the point in time that the erroneous data was introduced. Consider the following:

```py
class Graph:
  def __init__(self):
    self.nodes = []
    self.edge = []

  def add_node(self, node):
    self.nodes.append(node)

  def add_edge(self, edge):
    self.edge.append(edge)

class Node:
  def __init__(self, node_id):
    self.node_id = node_id

class Edge:
  def __init__(self, edge_id, from_node, to_node):
    self.edge_id = edge_id
    self.from_node = from_node
    self.to_node = to_node
```

The above class is very simple, but you could imagine a user may accidentally add an edge that references nodes not in the graph, like so:

```py
n1 = Node(1)
n2 = Node(2)
n3 = Node(2)
e1 = Edge(1, n1, n2)
e2 = Edge(2, n2, n3)
graph = Graph()
graph.add_node(n1)
graph.add_node(n2)
graph.add_edge(e1)
graph.add_edge(e2)
```

So now the above graph has a edge `e2` that references `n3` which isn't a member of `graph.nodes` and this might lead to errors. The way I've recently been solving this problem is to use the following.

```py
import functools


def decorator(func, validator):
    @functools.wraps(func)
    def new_method(cls_inst, *args, **kwargs):
        ret_val = func(cls_inst, *args, **kwargs)
        validator(cls_inst, *args, **kwargs)
        return ret_val
    return new_method


def add_validator(env='TESTING', validator):
    def class_decorator(cls):
        if env == 'TESTING':
            for attr, val in cls.__dict__.items():
                # attach validation function to all public methods:
                if callable(val) and not attr.startswith("_"):
                    setattr(cls, attr, decorator(val, validator))
        return cls
    return class_decorator


def validate_graph(graph, *args, **kwargs):
    for edge in graph.edges:
        if edge.from_node not in graph.nodes \
                or edge.to_node not in graph.nodes:
            raise ValueError('Connected node not in graph')

```

The above can be used to decorate the `graph` class and doing so will add the validator function to each of the public methods.

```py
@add_validator(env='TESTING', validator=validate_graph)
class Graph:
  def __init__(self):
    self.nodes = []
    self.edge = []

  def add_node(self, node):
    self.nodes.append(node)

  def add_edge(self, edge):
    self.edge.append(edge)
```

Now whenever a public method is called that puts the graph into an erroneous state an error will be raised, and it'll be raised in the place the error is introduced. This makes it easier to develop and once your done you can turn it off by changing an environmental variable you pass in instead of hard coding "TESTING".

__Why not just use validation?__

A valid criticism is that this is just validation and should be included in the methods themselves rather than adding programmatically via a decorator method.   

I think validation is to guard against incorrect end user input not developer misuse. The assumption is that the end code base should not allow methods to break the state of the objects you've defined. You should have already weeded out any such errors. This pattern aids to do said weeding by helping ensure that while the developer is writing code they know that there object states are always correct.

Also validation is hard. In general it's easier to change some data and check if it's in an invalid state than it is to check if some data will cause the object to enter into an invalid state and this pattern encourages this.

__In general:__

1. The main benefit is to ensure that an error is thrown the moment something happens that will later lead to a bug rather than when the bug actually happens.
2. Nicer to compartmentalize code into objects (classes) and allowed object states (validation functions) rather than combining them. This is like an superset of typing. Typing allows you to say this variable should be a certain thing, and if it's not then you throw an error. This pattern says this object should be in this state, i.e. all edges connect nodes in graph, and if it's not then you throw an error.
3. It forces each public method on the class to map between valid states. Which is a pretty strong condition but means developers will write good interfaces to their classes. A public method should define transitions on valid states otherwise the end user will need to know how and when to break and unbreak the object which seems bad. If the developer wants a method that moves the state of the object to an invalid state then they should do so using a private method.
4. Easier to achieve good coverage. Because it'll be included in the tests you write it will also ensure correct state of objects throughout which means a test that tests one thing could also be testing a tun of other things.

I think I have a couple of criticisms of it too.

1. Firstly it's not fun code to write and so it's often tempting to ignore it. This is like tests though and isn't a good reason to not use tests.
2. I've used it a fair amount in [this project](https://github.com/mauicv/gerel) but sometimes as a result of getting an error and then going all the way back to finding the point where the object state has been broken and _then_ writing it into the validator rather than the ideal where you describe all the allowed object states and then catch the errors _before_ they happen.

These being true however I feel development of [gerel](https://github.com/mauicv/gerel) did benefit from this pattern. [gerel](https://pypi.org/project/gerel/) is a simple evolutionary algorithm library. A lot of the objects are just list of numbers and it's basically impossible to know by eyeballing them if they're in the correct state or not.

![terminal output of large python object](/posts/state-validators/complex-object.png)

I'm not one hundred percent sure about this idea but It does appeal to the mathematical side of me. If your reading this and have any criticisms please email me to let me know! Also I have no idea if it's something lots of people already use under some name I don't know.
