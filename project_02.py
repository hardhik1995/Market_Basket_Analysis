#http://intelligentonlinetools.com/blog/2018/02/10/how-to-create-data-visualization-for-association-rules-in-data-mining/

dataset2 = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import random
import matplotlib.pyplot as plt
import numpy as np

oht = OnehotTransactions()
oht_ary = oht.fit(dataset2).transform(dataset2)
df2 = pd.DataFrame(oht_ary, columns=oht.columns_)
print (df2)

frequent_itemsets2 = apriori(df2, min_support=0.6, use_colnames=True)
print (frequent_itemsets2)


association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.7)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1.2)
print (rules2)



def draw_graph(rules, rules_to_show):
  import networkx as nx
  G1 = nx.DiGraph()

  color_map=[]
  N = 50
  colors = np.random.rand(N)
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']


  for i in range (rules_to_show):

      G1.add_nodes_from(["R"+str(i)])

      for a in rules2.iloc[i]['antecedents']:

          G1.add_nodes_from([a])

          G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

      for c in rules2.iloc[i]['consequents']:

          G1.add_nodes_from(c)

          G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)

  for node in G1:

      found_a_string = False
      for item in strs:
          if node==item:
               found_a_string = True
      if found_a_string:
          color_map.append('yellow')
      else:
          color_map.append('green')



  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]

  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)

  for p in pos:
      pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()

draw_graph (rules2, 6)
