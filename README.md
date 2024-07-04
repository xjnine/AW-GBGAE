# AW-GBGAE: An Adaptive Weighted Graph Auto-Encoder Based on Granular-Ball for General Data Clustering

Most existing graph neural network clustering algorithms are either limited to graph-type data and cannot be widely applied to clustering, or they can cluster ordinary data but require manual construction of graph structures and rely on the adjustment of many parameters. To overcome these limitations, we propose a new weighted graph generation and update method that combines multi-granularity representation and feature weight information. The core of this method lies in leveraging the excellent feature representation capabilities of weighted granular-balls and constructing the connections between these balls using a refined method, thereby capturing both local details and the global structure of the data more comprehensively. To maintain the accuracy and completeness of the graph results, a reconstruction-based update method that fully utilizes the weighted information from previous nodes is employed. By iteratively updating and reconstructing the graph structure, the intrinsic structure and attributes of the graph data are more accurately captured. Comprehensive experimental results demonstrate that the model (AW-GBGAE) performs well in clustering tasks and shows strong competitiveness compared to baseline models.

## How to run AW-GBGAE

```
python run.py
```

## Requirements

pytorch 1.3.1

scipy 1.3.1

scikit-learn 0.21.3

numpy 1.16.5