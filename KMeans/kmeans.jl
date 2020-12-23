using DelimitedFiles
using Plots
using Distributions
using InvertedIndices
using Printf

struct Point
    x::Float64
    y::Float64
end


mutable struct Cluster
    centroid::Point
    points::Array{Point,1}
end

mutable struct Classification
    clusters::Array{Cluster,1}
    internalEval::Float64
    externalEval::Float64
end

function euclidianDistance(a::Point, b::Point)
    return sqrt((a.x-b.x)^2 + (a.y - b.y)^2)
end

function createInitialClusters(dataPoints::Array{Point,1}, clusterNum::Int64)
    centroids, clusters = sample(points,clusterNum,replace=false), Cluster[]
    for centroid in centroids
        push!(clusters,Cluster(centroid,Point[]))
    end

    for point in dataPoints
        bestCluster = Inf
        bestDistance = Inf
        for (id,cluster) in enumerate(clusters)
            currDistance = euclidianDistance(point,cluster.centroid)
            if bestDistance > currDistance
                bestDistance = currDistance
                bestCluster = id
            end
        end
        push!(clusters[bestCluster].points,point)
    end

    return clusters
end


function plotClusters(clusters::Array{Cluster,1},colors::Array{Symbol,1})
    gr()
    myPlot = plot([],[],seriestype = :scatter, title = "K-Means")
    for (cluster,col) in zip(clusters,colors)
        clusterPointsX = map(point->point.x,cluster.points)
        clusterPointsY = map(point->point.y,cluster.points)
        plot!(myPlot,clusterPointsX,clusterPointsY,color=[col],seriestype = :scatter)
        plot!([cluster.centroid.x],[cluster.centroid.y],color=[col],shape=[:star5], markersize=10, seriestype = :scatter)
    end
    display(myPlot)
end

#Davies–Bouldin index
function calcInternalEval(clusters::Array{Cluster,1})
    avgDistances = Float64[]
    for cluster in clusters
        avgDist, allPoints = 0.0, length(cluster.points)
        for point in cluster.points
            avgDist += euclidianDistance(point,cluster.centroid) / allPoints
        end
        push!(avgDistances,avgDist)
    end

    helper(clusterId,clusters,avgDistances) =
        maximum(j -> j == clusterId ? 0 :
                     (avgDistances[clusterId]+avgDistances[j]) /
                     euclidianDistance(clusters[clusterId].centroid,clusters[j].centroid),
                1:length(clusters))
    internalEval = sum(clusterId -> helper(clusterId,clusters,avgDistances),1:length(clusters))
    return internalEval / length(clusters)
end

#TODO to implement the external metric
#function updateExternalEval!(clusterId::Int64,clusters::Array{Cluster,1})

#end

function updateCentroids!(clusters::Array{Cluster,1})
    for i in 1:length(clusters)
        if length(clusters[i].points) == 0
            return
        end
        meanX = sum(point -> point.x, clusters[i].points) / length(clusters[i].points)
        meanY = sum(point -> point.y, clusters[i].points) / length(clusters[i].points)
        clusters[i].centroid = Point(meanX,meanY)
    end
end

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
        bestDistance = euclidianDistance(point,clusters[clusterId].centroid)
        bestCluster = clusterId
        for i in 1:length(clusters)
            if i == clusterId
                continue
            end

            competitorDistance =  euclidianDistance(point,clusters[i].centroid)
            if bestDistance > competitorDistance
                bestCluster = i
                bestDistance = competitorDistance
            end
        end

        if bestCluster != clusterId
            changes+=1
        end

        push!(clusters[bestCluster].points,point)
    end
    return changes
end


coords = readdlm("./data/unbalance/unbalance.txt", ' ', Int64, '\n')

points = Point[]
for coord in eachrow(coords)
    push!(points,Point(1.0*coord[1],1.0*coord[2]))
end

colors = [:blue,:green,:red, :yellow,:black, :purple, :orange,:brown, :gray]

bestAllTimeClassification = Classification(Cluster[],Inf,Inf)
for i in 2:9
    @printf("ClusterNumbers %d\n",i)
    @printf("ClusterNumbers %d\n",i)
    tries=100
    bestClassification = Classification(Cluster[],Inf,Inf)
    while tries > 0
        clusters = createInitialClusters(points,i)
        classJob = Classification(clusters,Inf,Inf)
        numChanges, iter, maxIter = Inf, 0, 400

        while numChanges > 0 ||  iter < maxIter
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
