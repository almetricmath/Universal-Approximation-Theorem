__author__ = 'Al Bernstein'
__license__ = 'MIT License'

# Draw Neural Net and output to .svg file

from graphviz import Digraph

g = Digraph('G', format='svg', filename='neuralnet.svg')
g.node('a', 'Input x')
g.node('b', '<h<sub>1</sub>(x + b<sub>1</sub>)>')
g.node('c', '<h<sub>2</sub>(x + b<sub>2</sub>)>')
g.node('d', '<&#8942;>')
g.node('e', '<h<sub>n</sub>(x + b<sub>n</sub>)>')
g.node('f', '< &#931;>')
g.node('g', 'output y(x)')
g.node('h',  '<y<sub>1</sub>>')
g.edge('a', 'b')
g.edge('a', 'c')
g.edge('a', 'd')
g.edge('a', 'e')
g.edge('b', 'f', xlabel = '<w<sub>1</sub>>')
g.edge('c', 'f', xlabel = '<w<sub>2</sub>>')
g.edge('e', 'f', xlabel = '<w<sub>n</sub>>')
g.edge('h', 'f')
g.edge('f', 'g')
g.graph_attr.update(rankdir='LR')
g.view()

