using DataFrames
using DataFramesMeta
using CSV
using Printf
using Plots
using Distributions
using Random
using InvertedIndices

mutable struct ID3
    feature::Int64
    children::Array{ID3,1}
    dominatingClass::String
    isLeaf::Bool
end

#we assume that the DataFrame arg is already filtered suck that it already uses all previously selected features
#length(foldr(((col,val),res)-> res .& df[:,col] .== val,usedFeatures))

#TODO use laplace smoothing

function entropyOneAttribute(df::DataFrame,classDomain::Array{String,1}; α=0.005)
    allCounts = nrow(df)
    #propbabilities = map(class -> (count(df[:,1] .== class) + α) /(allCounts + α * length(classDomain[1])) ,classDomain)
    propbabilities = map(class -> count(df[:,1] .== class)/allCounts ,classDomain)
    return - foldr((prob, res) -> prob == 0.0 ? res : res + prob * log2(prob) ,propbabilities,init=0.0)
end

function entropyTwoAttributes(df::DataFrame,feature::Int64,featureDomain::Array{String,1},classDomain::Array{String,1}; α = 0.005)
    allCounts, ∑ = nrow(df), 0.0
    for featureValue in featureDomain
        entropy = entropyOneAttribute( df[df[:,feature] .== featureValue,:],classDomain)
        #propOfFeature = (count(df[:,feature] .== featureValue) + α) / (allCounts + α * length(unique(df[:,feature])))
        propOfFeature = count(df[:,feature] .== featureValue)/ allCounts
        @printf("Feature %d with value %s has entropy %.6f and prob %.6f\n",feature,featureValue,entropy,propOfFeature)
        ∑ += entropy * propOfFeature
    end
    return ∑
end

function calcGain(currentEntropy,parentEntropy)
    return parentEntropy - currentEntropy
end

function getDominatingClass(df::DataFrame, classDomain::Array{String,1}; parentDominatingClass = "Sentinel" )
    winners = []
    dominatingCount = 0
    for class in classDomain
        cnt = count(df[:,1] .== class)
        if dominatingCount < cnt
            dominatingCount = cnt
            winners = [class]
        elseif dominatingCount == cnt
            push!(winners,class)
        end
    end

    if length(winners) == 1
        @printf("Winner is %s\n",winners[1])
        @printf("Winner is %s\n",winners[1])
        return winners[1]
    elseif parentDominatingClass != "Sentinel" && parentDominatingClass in winner
        @printf("Winner is %s\n",parentDominatingClass)
        @printf("Winner is %s\n",parentDominatingClass)
        return parentDominatingClass
    end


    return winners[(rand(Int) % length(winners))+1]
end

function getFeatureWithBestGain(df::DataFrame, domains::Array{Array{String,1},1},currEntropy::Float64)
    winner, maxGain = 2, -Inf
    println(size(domains,1))
    println(domains)
    for i in 2:size(domains,1)
        gain = calcGain(entropyTwoAttributes(df,i,domains[i],domains[1]),currEntropy)
        if maxGain < gain
            maxGain = gain
            winner = i
        end
    end
    return winner
end

function createID3(df::DataFrame,domains::Array{Array{String,1},1};setThreshold=5)
    println("------------")
    println(domains)
    println(df)
    dominatingClass = getDominatingClass(df,domains[1])
    currEntropy = entropyOneAttribute(df,domains[1])

    winner = getFeatureWithBestGain(df,domains,currEntropy)
    root = ID3(winner,ID3[],dominatingClass,false)

    for featureVal in domains[winner]
        filteredSet = df[df[:,winner] .== featureVal,:]
        child = createID3Recursive(filteredSet,domains[Not(winner)],setThreshold,dominatingClass)
        push!(root.children, child)
    end
    return root
end

function isHomogenous(currEntopy::Float64; ϵ=0.03)
    return abs(currEntopy - 1.0) <= ϵ || currEntopy <= ϵ
end

function createID3Recursive(df::DataFrame,domains::Array{Array{String,1},1},setThreshold, parentDominatingClass::String)
    dominatingClass = getDominatingClass(df,domains[1],parentDominatingClass = parentDominatingClass)
    currEntropy =  entropyOneAttribute(df,domains[1])
    if nrow(df) <= setThreshold || length(domains) == 0 || isHomogenous(currEntropy)
        return ID3(-1,ID3[],dominatingClass,true)
    end

    winner = getFeatureWithBestGain(df,domains,currEntropy)
    root = ID3(winner,ID3[])

    for featureVal in domains[winner]
        filteredSet = df[df[:,winner] .== featureVal]
        child = createID3Recursive(filteredSet,domains[Not(winner)],setThreshold,dominatingClass)
        push!(root.children, child)
    end
    return root
end

function pickValue(valueProbabilities::Array{Float64,1}, domain::Any)
    result = rand(Multinomial(1, valueProbabilities))
    return domain[findfirst(x -> x == 1, result)]
end

function fillValues!(df::DataFrame, domains::Array{Array{String,1},1})
    for col in 1:ncol(df)
        distribution = map(val -> count(df[:,col] .== val), domains[col])
        probDistribution = 1.0 * distribution ./ sum(distribution)
        missingValueRows = foldr((row, res) -> df[row,col] == "?" ? push!(res,row) : res, 1:nrow(df), init=[])

        for row in missingValueRows
            df[row,col] = pickValue(probDistribution,domains[col])
        end
    end
end

#df = CSV.read("./data/breast-cancer.data",header=false)

#const missingValuesThreshold = 1.0 * size(df, 2) / 2
#const missingValueSign = "?"

#for i in 1:size(df,2)
#    println(count(row -> df[row,i] == missingValueSign,1:size(df,1)))
#end

#or i in 1:size(df,1)
#    cnt = count(col -> df[i,col] == missingValueSign,1:size(df,2))
#    if cnt > 0
#        @printf("Row index %d and missing values %d\n",i,cnt)
#    end
#end

#domains = [["no-recurrence-events", "recurrence-events"],
#           ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
#           ["lt40", "ge40", "premeno"],
#           ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
#                         "45-49", "50-54", "55-59"],
#           ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26",
#                        "27-29", "30-32", "33-35", "36-39"],
#           ["yes", "no"],
#           ["1", "2", "3"],
#           ["left", "right"],
#           ["left-up", "left-low", "right-up", "right-low", "central"],
#           ["yes", "no"]]


df = CSV.read("golf.csv",header=false,type=String)

domains = [["Yes","No"],
           ["Rainy","Overcast","Sunny"],
           ["Hot","Mild","Cool"],
           ["High","Normal"],
           ["True","False"]]

permutecols!(df, [:Column5, :Column1,:Column2,:Column3,:Column4])


#entropyTwoAttributes(df,2,domains[2],domains[1])
createID3(df,domains)
