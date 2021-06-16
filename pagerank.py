# Run this script in Spark 2.4.5 for GraphFrame functionality.
 
# Load data
data = spark.read.option("header", "true").csv("/FileStore/tables/207146740_T_T100D_SEGMENT_US_CARRIER_ONLY.csv")
 
# Create edges table
origin_and_dest = data.select("ORIGIN", "DEST").toDF("src", "dst")  #removes extra spurious column
origin_and_dest.show()


from pyspark.sql.functions import lit
from graphframes import GraphFrame
 
# Create vertices table
airport_names = origin_and_dest.select("src").distinct()
default_rank = 10  # Alternatively, set equal to 1/airport_names.count() to have the sum of all ranks be 1.
airports = airport_names.withColumnRenamed("src", "id").withColumn("rank", lit(default_rank))
airports.show()
 
# Create GraphFrame
airports_graph = GraphFrame(airports, origin_and_dest)

from pyspark.sql.functions import col
 
def pagerank(graph: GraphFrame, current_iteration: int = 0):
  """Finds PageRank for each vertex in graph after 5 iterations. Vertices must have attributes id and rank"""
  
  # Exiting condition from recursion
  if current_iteration == 5:  # Iteration count can be set higher, although it will cause the function to take much longer to run.
    return graph
  
  # Define values used in PageRank formula
  alpha = 0.15
  total_vertices = graph.vertices.count()
  
  # Join tables and calculate new ranks
  joined1 = graph.edges.join(graph.vertices, graph.edges.src == graph.vertices.id, how="inner")
  joined2 = joined1.join(graph.outDegrees, joined1.src == graph.outDegrees.id, how="inner")
  sigma = joined2.withColumn("sigma_n", col("rank") / col("outDegree")).groupBy("dst").sum("sigma_n")
  new_vertices = sigma.withColumn("new_rank", alpha * (1 / total_vertices) + (1 - alpha) * col("sum(sigma_n)")) # PageRank formula
  
  # Reformat columns to match original vertices columns "id" and "rank" 
  formatted_vertices = new_vertices.select("dst", "new_rank").withColumnRenamed("dst", "id").withColumnRenamed("new_rank", "rank")
  new_graph = GraphFrame(formatted_vertices, graph.edges)
  
  # Release memory of unused variables
  del joined1, joined2, sigma, new_vertices, formatted_vertices
  
  # Recusively call pagerank()
  return pagerank(new_graph, current_iteration + 1)
 
# Call pagerank() and display top 10 ranked airports
airport_rankings = pagerank(airports_graph)
airport_rankings.vertices.orderBy(col("rank").desc()).limit(10).show()
