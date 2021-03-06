__author__ = 'Al Bernstein'
__license__ = 'MIT License'

# Draw Neural Net

from graphviz import Digraph

g = Digraph('G', format='svg', filename='neuralnet_2.svg')
g.node('a', 'Input x')
g.node('b',  '<b<sub>1</sub>>')
g.node('c', '< &#931;>')
g.node('d', '<SEU(-1, x + b<sub>1</sub>)>')
g.node('e', '<y<sub>1</sub>(x)>')
g.node('f', '<SEU(1, y<sub>1</sub>(x))>')
g.node('g', '<output y<sub>2</sub>(x)>')


g.edge('a', 'c')
g.edge('b', 'c')
g.edge('c', 'd', xlabel = '<&alpha;<sub>1</sub>>')
g.edge('d', 'e')
g.edge('e', 'f', xlabel = '<&omega;<sub>1</sub>>')
g.edge('f', 'g')

g.graph_attr.update(rankdir='LR')
g.view()

