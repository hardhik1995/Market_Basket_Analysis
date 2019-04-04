#https://github.com/chris1610/pbpython/blob/master/notebooks/Market_Basket_Intro.ipynb
#https://www.analyticsvidhya.com/blog/2014/08/visualizing-market-basket-analysis/
#http://intelligentonlinetools.com/blog/2018/02/10/how-to-create-data-visualization-for-association-rules-in-data-mining/

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df.head()

# Clean up spaces in description and remove any rows that don't have a valid invoice
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)


df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]


basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


basket.head()


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

print(basket_sets)

"""
# calculate occurrence(support) for every product in all transactions
product_support_dict = {}
for column in basket_sets.columns:
    product_support_dict[column] = sum(basket_sets[column]>0)

# visualise support
pd.Series(product_support_dict).plot(kind="bar")
"""

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Advanced and strategical data frequent set selection
frequent_itemsets[ (frequent_itemsets['length'] > 1) &
                   (frequent_itemsets['support'] >= 0.02) ].head()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

rules.sort_values('lift', ascending=False).head()


print(rules[(rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8)])

print(len(rules))



# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(14, 8))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], alpha=0.9, cmap='YlOrRd')
plt.title('Rules distribution color mapped by lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar()

#####
# Visualizing the rules

support=rules.as_matrix(columns=['support'])
confidence=rules.as_matrix(columns=['confidence'])

for i in range (len(support)):
    support[i] = support[i]
    confidence[i] = confidence[i]

plt.title('Association Rules')
plt.xlabel('support')
plt.ylabel('confidence')
sns.regplot(x=support, y=confidence, fit_reg=False)

plt.gcf().clear()

###########
import networkx as nx
import numpy as np

def draw_graph(rules, rules_to_show):
  import networkx as nx
  G1 = nx.DiGraph()

  color_map=[]
  N = 50
  colors = np.random.rand(N)
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']


  for i in range (rules_to_show):

      G1.add_nodes_from(["R"+str(i)])

      for a in rules.iloc[i]['antecedents']:

          G1.add_nodes_from([a])

          G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

      for c in rules.iloc[i]['consequents']:

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

draw_graph (rules, 10)

