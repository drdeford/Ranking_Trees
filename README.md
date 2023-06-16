# Python code and Jupyter notebooks for *Ranking Trees Based on Global Centrality Measures*

Trees, or connected graphs with no cycles, are a commonly studied combinatorial family, and the star 
graph and the path of a fixed order $n$ frequently provide extremal values for 
natural metrics on networks and graphs. In the paper *Ranking Trees Based on Global Centrality Measures*, 
we prove inequalities for several global centrality measures, such as global closeness and betweenness centralities, 
and for graphical Stirling numbers of the first kind that interpolate these extremes. 
Moreover, we provide two algorithms that allow us to traverse the space of non-isomorphic 
trees of a fixed order, one towards the star graph of the same order and the other towards the path. 
Furthermore, we investigate the relationship between these global centrality measures on the one hand
and the $(n-2)$-nd Stirling numbers of the first kind for small trees on the other hand, demonstrating a strong association between them, in particular with respect 
to the partial orderings obtained from applying our two interpolating algorithms. Based on our observations from these small trees, 
we prove general bounds that relate the $(n-2)$nd Stirling numbers of the first kind of trees of order $n$ to these global centrality measures. Finally, we provide two related approaches to totally order the set of all non-isomorphic trees of fixed order. We show that the totally ordering
obtained from one of these approaches is consistent with the poset structure obtained from our two 
tree interpolation algorithms in addition to being one of the features to use for predicting the $(n-2)$-nd Stirling numbers of the first kind for small trees.
The Python code here helps us explore some of these properties computationally.
 

## **Ranking\_Trees--Centrality\_Bounds\_1.0.ipynb** 

The poset structures on the set of non-isomorphic unlabelled trees based on path-to-star and star-to-path algorithms are explored in this notebook. 

<center>
  <table>
    <tr>
      <td><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/Figures/path_to_star_11.png' width = '400'>
      <br>The poset structure for trees of order 11 based on the path-to-star algorithm
      </td>
      <td><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/Figures/star_to_path_11.png' width = '400'>
      <br>The poset structure for trees of order 11 based on the star-to-path algorithm
      </td>
    </tr>
  </table>
</center>
<center>
  <table>
    <tr>
      <td><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/Figures/trees_clo_11.png' width = '400'>
      <br>The scatterplot for the association between global closeness centrality and $(n-2)^{\text{nd}}$ Stirling number of the first kind for trees of order $11$
      </td>
      <td><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/Figures/trees_bet_11.png' width = '400'>
      <br>The scatterplot for the association between global betweenness centrality and $(n-2)^{\text{nd}}$ Stirling number of the first kind for trees of order $11$
      </td>
    </tr>
  </table>
</center>

## **Ranking\_Trees--Checking\_Bounds\_1.0.ipynb**

Some of the results regarding closeness and betweenness centralities for small trees are validated numerically in this notebook. 

## **Ranking\_Trees--Comparing\_Total\_Orderings\_1.0.ipynb**

This notebook contains the code for comparing total orderings obtained from the distinguishing polynomial using two different approaches (degree-based and evaluation-based).

<center>Comparing degree-based and evaluation-based orderings</center>
<center><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/Figures/ranking_total.gif' width = '400'></center>
<br>

<centre>Comparing path-to-star-based and evaluation-based orderings<br></center>
<center><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/Figures/ranking_path_to_star.gif' width = '400'></center>
<br>

<center>Comparing star-to-path-based and evaluation-based orderings</center>
<center><img src = 'https://github.com/drdeford/Ranking_Trees/blob/main/Figures/ranking_star_to_path.gif' width = '400'></center>
<br>

## **Ranking\_Trees--Distinguishing\_Polynomials\_1.0.ipynb**

This notebook contains the code for computing the total orderings obtained from the distinguishing polynomial using two different approaches (degree-based and evaluation-based) and for exploring the connections between these total orderings and other graph statistics discussed in the paper.

## **Ranking\_Trees--Small\_Examples\_1.0.ipynb**

This notebook contains the explanatory data visualizations for small trees.

## **tree\_functions_1.py** 

All the functions defined by the authors and used in the above notebooks are in this file.  
