import random

import matplotlib.pyplot as plt
import networkx as nx


G=nx.cycle_graph(24)
pos=nx.spring_layout(G, iterations=200)
# nx.draw(G,pos,node_color=range(24), node_size=800, cmap=plt.cmapm.Blues)
colors = [random.randrange(0, 100) / 100 for i in range(24)]
nx.draw(G, node_color=colors)
# plt.savefig("node_colormap.png") # save as png
plt.show() # display