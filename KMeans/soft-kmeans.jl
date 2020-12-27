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

#euclidian distance
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

colors = [UInt8[	191, 255, 0],
          UInt8[	238, 130, 238],
          UInt8[	220, 20, 60],
          UInt8[	255, 215, 0],
          UInt8[	148, 0, 211],
          UInt8[	0, 191, 255],
          UInt8[    255, 165, 0],
          UInt8[	0, 255, 255],
          UInt8[	128, 128, 128]]

for i in 1:8
    @printf("ClusterNumbers %d\n",i)
    tries=5
    while tries > 0
        classJob = createInitialClassification(points,i)
        numChanges, iter, maxIter = Inf, 0, 100

        while numChanges != 0 &&  iter < maxIter
            updateCentroids!(classJob)
            numChanges = updateMembershipProb!(classJob)
            iter+=1
        end
        tries-=1
        plotClassification(classJob,colors)
    end
end

#TODO internal metric to implement
