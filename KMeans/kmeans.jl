using DelimitedFiles
using Plots
using Distributions
using InvertedIndices
using Printf
using StatsBase

#representation of 2D point
struct Point
    x::Float64
    y::Float64
end

#structure representing a cluster
mutable struct Cluster
    centroid::Point
    points::Array{Point,1}
end

#structure representing the classification job overall
mutable struct Classification
    clusters::Array{Cluster,1}
    internalEval::Float64
    externalEval::Float64
end

#Euclidian distance
function dist(a::Point, b::Point)
    return sqrt((a.x-b.x)^2 + (a.y - b.y)^2)
end

#generate initial centroids with kmeans++ optimization
#returns the position of the centroids
function generateCentroids(dataPoints::Array{Point,1}, clusterNum::Int64)
    remainingPoints = [i for i in 1:length(dataPoints)]
    distances = Float64[Inf for _ in 1:length(dataPoints)]

    centroids = [dataPoints[sample(remainingPoints)]]
    for i in 1:length(dataPoints)
        distances[i] = dist(centroids[1],dataPoints[i]) ^ 2
    end

    for i in 2:clusterNum
        newCentroid = dataPoints[sample(remainingPoints,Weights(distances))]
        for i in 1:length(dataPoints)
            distances[i] = min(dist(newCentroid,dataPoints[i]) ^ 2,distances[i])
        end
        push!(centroids,newCentroid)
    end

    return centroids
end

#generates clusters
#returns the clusters
function createInitialClusters(dataPoints::Array{Point,1}, clusterNum::Int64)
    centroids = generateCentroids(dataPoints,clusterNum)
    clusters = map(c -> Cluster(c,Point[]),centroids)

    for point in dataPoints
        winner = foldr((x,res) -> res == () || x[2] < res[2] ? x : res,
              map(id -> (id,dist(point,clusters[id].centroid)), 1:length(clusters)),
              init = ())

        push!(clusters[winner[1]].points,point)
    end
    return clusters
end

#plotting the classification results
function plotClusters(clusters::Array{Cluster,1},colors::Array{Symbol,1})
    gr()
    myPlot = plot([],[],seriestype = :scatter, title = "K-Means", legend=false)
    for (cluster,col) in zip(clusters,colors)
        clusterPointsX = map(point->point.x,cluster.points)
        clusterPointsY = map(point->point.y,cluster.points)
        plot!(clusterPointsX,clusterPointsY,color=[col],seriestype = :scatter)
        plot!([cluster.centroid.x],[cluster.centroid.y],color=[col],shape=[:star5], markersize=10, seriestype = :scatter)
    end
    display(myPlot)
end

#Davies–Bouldin index - used as a measurement for the best number of clusters
#returns Davies–Bouldin index
function calcInternalEval(clusters::Array{Cluster,1})
    avgDists = map(cl ->
        sum(p -> dist(p,cl.centroid), cl.points)/length(cl.points),
        clusters)

    helper(id,cls,avgDists) =
        maximum(j -> (avgDists[id]+avgDists[j]) /
                     dist(cls[id].centroid,cls[j].centroid),
                [i for i in 1:length(cls) if i != id])

    internalEval = sum(clId -> helper(clId,clusters,avgDists),1:length(clusters))
    return internalEval / length(clusters)
end

#calculate the position of the centroid in the cluster
function updateCentroids!(clusters::Array{Cluster,1})
    for i in 1:length(clusters)
        if length(clusters[i].points) == 0
            break
        end
        meanX = sum(point -> point.x, clusters[i].points) / length(clusters[i].points)
        meanY = sum(point -> point.y, clusters[i].points) / length(clusters[i].points)
        clusters[i].centroid = Point(meanX,meanY)
    end
end

#updates the cluster -> position of centroids, points in each cluster
#returns the number of swaps of points from one cluster to another
function updateClusters!(clusters::Array{Cluster,1})
    updateCentroids!(clusters)
    points = []
    for i in 1:length(clusters)
        for point in clusters[i].points
            push!(points,(point,i))
        end
        clusters[i].points = Point[]
    end

    changes = 0
    for (point,clusterId) in points
        #winner -> (clusterId, closest distance to the centroid)
        winner = foldr((x,res) -> res == () || x[2] < res[2] ? x : res,
              map(id -> (id, dist(point,clusters[id].centroid)), 1:length(clusters)),
              init = ())

        changes += (winner[1] != clusterId ? 1 : 0)
        push!(clusters[winner[1]].points,point)
    end

    return changes
end


coords = readdlm("./data/unbalance/unbalance.txt", ' ', Int64, '\n')

points = Point[]
for coord in eachrow(coords)
    push!(points,Point(1.0*coord[1],1.0*coord[2]))
end

colors = [:violet,:lime,:crimson, :gold,:darkviolet, :deepskyblue, :orange,:aqua, :gray]

bestAllTimeClassification = Classification(Cluster[],Inf,Inf)
for i in 2:8
    @printf("ClusterNumbers %d\n",i)
    tries=5
    bestClassification = Classification(Cluster[],Inf,Inf)
    while tries > 0
        clusters = createInitialClusters(points,i)
        classJob = Classification(clusters,Inf,Inf)
        numChanges, iter, maxIter = Inf, 0, 100

        while numChanges > 0 &&  iter < maxIter
            numChanges = updateClusters!(classJob.clusters)
            classJob.internalEval = calcInternalEval(classJob.clusters)
            iter+=1
        end

        if bestClassification.internalEval > classJob.internalEval
            bestClassification = classJob
        end
        tries-=1
    end

    @printf("Classification was with %d number of clusters and internal eval %.6f\n",
        length(bestClassification.clusters),bestClassification.internalEval)

    if bestAllTimeClassification.internalEval > bestClassification.internalEval
        bestAllTimeClassification = bestClassification
    end

    plotClusters(bestClassification.clusters,colors)
end

@printf("Best classification was with %d number of clusters and internal eval %.6f\n",
    length(bestAllTimeClassification.clusters),bestAllTimeClassification.internalEval)
