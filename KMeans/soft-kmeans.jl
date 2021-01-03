using DelimitedFiles
using Plots
using Distributions
using InvertedIndices
using Printf
using StatsBase

mutable struct Point
    x::Float64
    y::Float64
end

#structure representing the classification job overall
mutable struct Classification
    centroids::Array{Point,1}
    points::Array{Point,1}
    memProbTable::Array{Array{Float64,1},1}
    internalEval::Float64
end

#Euclidian distance
function dist(a::Point, b::Point)
    return sqrt(squaredDist(a,b))
end

function squaredDist(a::Point, b::Point)
    return (a.x-b.x)^2 + (a.y - b.y)^2
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

#β is stiffness parameter
#updates the table for membership probability of each point
function updateMembershipProb!(class::Classification; β = 10)
    changes = 0
    for i in 1:length(class.points)
        ∑ = 0.0
        oldProbs = copy(class.memProbTable[i])
        for j in 1:length(class.centroids)
            class.memProbTable[i][j] = ℯ^(-10*dist(class.points[i],class.centroids[j]))
            ∑ += class.memProbTable[i][j]
        end
        if oldProbs != class.memProbTable[i]
            changes+=1
        end
        class.memProbTable[i] = class.memProbTable[i] / ∑ #normalization of the "vector"
    end
    return changes
end

function calcStd(centroid,points)
    return sqrt(sum(p->squaredDist(p,centroid),points)/length(points))
end


#Soft Davies-Bouldin Index - used as a measurement for the best number of clusters
function calcInternalEval(classJob::Classification)
    avgDists = map(c -> calcStd(c,classJob.points),classJob.centroids)
    avgMembership = map(cID-> sum(classJob.memProbTable[:][cID])/length(classJob.points),1:length(classJob.centroids))

    helper(id,centroids,avgDists, avgMembership) =
        maximum(j -> (avgDists[id]*avgMembership[id]+avgDists[j]*avgMembership[j]) /
                     dist(centroids[id],centroids[j]),
                [j for j in 1:length(centroids) if j != id])

    internalEval = sum(clId -> helper(clId,classJob.centroids,avgDists,avgMembership),1:length(classJob.centroids))
    return internalEval / length(classJob.centroids)
end

#generates clusters
#returns the clusters
function createInitialClassification(dataPoints::Array{Point,1}, clusterNum::Int64)
    centroids = generateCentroids(dataPoints,clusterNum)
    classification = Classification(centroids,dataPoints,[[0.0 for _ in 1:length(centroids)] for _ in 1:length(dataPoints)],0.0)
    updateMembershipProb!(classification)
    return classification
end

#calculate the position of the centroid in the cluster
function updateCentroids!(classJob::Classification)
    points, memProbTable = classJob.points, classJob.memProbTable
    for i in 1:length(classJob.centroids)
        weightedMeanX = sum(id -> memProbTable[id][i] * points[id].x,1:length(points)) /
            sum(id -> memProbTable[id][i],1:length(points))
        weightedMeanY = sum(id -> memProbTable[id][i] * points[id].y,1:length(points)) /
            sum(id -> memProbTable[id][i],1:length(points))
        classJob.centroids[i] = Point(weightedMeanX,weightedMeanY)
    end
end

function plotClassification(classJob::Classification,colors::Array{Array{UInt8,1},1})
    pointColors = []
    for i in 1:length(classJob.points)
        acc = UInt8[0,0,0]
        for j in 1:length(classJob.centroids)
            for k in 1:length(colors[j])
                acc[k]+=convert(UInt8,round(colors[j][k] * classJob.memProbTable[i][j],digits=0))
            end
        end
        push!(pointColors,RGB(acc/255...))
    end

    gr()
    pointsX = map(point->point.x,classJob.points)
    pointsY = map(point->point.y,classJob.points)
    myPlot = plot(pointsX,pointsY,color=pointColors,seriestype = :scatter, title = "K-Means", legend=false)

    centroidsX = map(point->point.x,classJob.centroids)
    centroidsY = map(point->point.y,classJob.centroids)
    plot!(centroidsX,centroidsY,color=[RGB(color/255...) for color in colors],seriestype = :scatter,shape=[:star5], markersize=10)
    display(myPlot)
end


coords = readdlm("./data/normal/normal.txt", '\t', Float64, '\n')

points = Point[]
for coord in eachrow(coords)
    push!(points,Point(1.0*coord[1],1.0*coord[2]))
end

colors = [ UInt8[	0, 255, 255],
          UInt8[	191, 255, 0],
          UInt8[	238, 130, 238],
          UInt8[	220, 20, 60],
          UInt8[	255, 215, 0],
          UInt8[	148, 0, 211],
          UInt8[	0, 191, 255],
          UInt8[    255, 165, 0],
          UInt8[	128, 128, 128]]


bestAllTimeClassification = Classification([],[],[],Inf)
for i in 2:8
    @printf("ClusterNumbers %d\n",i)
    tries=5
    bestClassification = Classification([],[],[],Inf)
    while tries > 0
        classJob = createInitialClassification(points,i)
        numChanges, iter, maxIter = Inf, 0, 100

        while numChanges != 0 &&  iter < maxIter
            updateCentroids!(classJob)
            numChanges = updateMembershipProb!(classJob)
            classJob.internalEval = calcInternalEval(classJob)
            iter+=1
        end
        tries-=1
        if bestClassification.internalEval > classJob.internalEval
            bestClassification = classJob
        end
    end

    @printf("Classification was with %d number of clusters and internal eval %.6f\n",
        length(bestClassification.centroids),bestClassification.internalEval)

    if bestAllTimeClassification.internalEval > bestClassification.internalEval
        bestAllTimeClassification = bestClassification
    end

    plotClassification(bestClassification,colors)
end

@printf("Best classification was with %d number of clusters and internal eval %.6f\n",
    length(bestAllTimeClassification.centroids),bestAllTimeClassification.internalEval)
