# Python code and Jupyter notebooks for *Ranking Trees Based on Global Centrality Measures*

Trees, or connected graphs with no cycles, are a commonly studied combinatorial family, and the star 
graph and path graph of a fixed size are frequently extremal values for natural metrics on networks 
and graphs. In the paper *Ranking Trees Based on Global Centrality Measures*, 
we prove several monotonicity results for global centrality measures and 
Stirling numbers that interpolate these extremes. The Python code here helps us explore
some of these properties computationally. 

## **Ranking\_Trees--Centrality\_Bounds\_1.0.ipynb** 

The poset structures on the set of non-isomorphic unlabelled trees based on path-to-star and star-to-path algorithms are explored in this notebook. 

<center><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/path_to_star_11.png' width = '400'> The poset structure for trees of order 11 based on the path-to-star algorithm</center>

<center><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/star_to_path_11.png' width = '400'>The poset structure for trees of order 11 based on the star-to-path algorithm</center>

## **Ranking\_Trees--Checking\_Bounds\_1.0.ipynb**

Some of the results regarding closeness and betweenness centralities for small trees are validated numerically in this notebook. 

## **Ranking\_Trees--Comparing\_Total\_Orderings\_1.0.ipynb**

Contains the code for comparing total orderings obtained from the distinguishing polynomial using two different approaches (degree-based and evaluation-based).

<center><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/ranking_total.gif' width = '400'></center>

## **Ranking\_Trees--Distinguishing\_Polynomials\_1.0.ipynb**

Contains the code for computing the total orderings obtained from the distinguishing polynomial using two different approaches (degree-based and evaluation-based) and for exploring the connections between these total orderings and other graph statistics discussed in the paper.

## **Ranking\_Trees--Small\_Examples\_1.0.ipynb**

Contains the explanatory data visualizations for small trees.

## **tree\_functions_1.py** All the functions defined by the authors and used in the above notebooks are in this file.  
