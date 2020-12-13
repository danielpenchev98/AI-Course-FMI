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
    class::String
end

#we assume that the DataFrame arg is already filtered suck that it already uses all previously selected features
#length(foldr(((col,val),res)-> res .& df[:,col] .== val,usedFeatures))

#TODO use laplace smoothing

function entropyOneAttribute(df::DataFrame,classDomain::Array{String,1})
    allCounts = nrow(df)
    propbabilities = map(class -> count(df[:,1] .== class)/allCounts ,classDomain)
    println(propbabilities)
    propbabilities = ["sfa"]
    return - foldr((prob, res) -> res + prob * log2(prob) ,propbabilities,init=0.0)
end

function entropyTwoAttributes(df::DataFrame,feature::Int64,featureDomain::Array{String,1},classDomain::Array{String,1})
    allCounts = nrow(df)

    ∑ = 0

    for featureValue in featureDomain
        enrop = entropyOneAttribute( df[df[:,feature] .== featureValue,:],classDomain)
        propOfFeature = length(df[:,feature] .== featureValue)/allCounts
        @printf("Feature %d with value %s has single entropy %.6f abd propOfFeature %.6f\n",feature,featureValue,enrop,propOfFeature)
        ∑ += enrop * propOfFeature
    end

    return ∑
    #return sum([entropyOneAttribute( df[df[:,feature] .== featureValue,:],classDomain)
    #           * length(df[:,feature] .== featureValue)/allCounts for featureValue in featureDomain])
end
function calcGain(currentEnropy,parentEntropy)
    if isempty(parentEntropy)
        return currentEnropy
    end
    return parentEntropy - currentEnropy
end

# P(x|C,y) = P(x,C,y)/P(C,y)

function createID3(df::DataFrame,domains::Array{Array{String,1},1};setThreshold=5)
    maxGain = 0
    winner = 2
    for i in 2:size(domains,1)
        gain = calcGain(entropyTwoAttributes(df,i,domains[i],domains[1]),[])
        if maxGain < gain
            maxGain = gain
            winner = i
        end
    end

    root = ID3(winner,ID3[],"None")
    for featureVal in domains[winner]
        push!(root.children,createID3Recursive(df[df[:,winner] .== featureVal],domains[Not(winner)],setThreshold))
    end
    return root
end

function creatID3Recursive(df::DataFrame,domains::Array{Array{String,1},1},setThreshold)
    if nrow(df) >= setThreshold || length(domains) == 0

        return ID3(-1,ID3[],"")
    end

    maxGain = 0
    winner = 2
    for i in 2:size(domains,1)
        gain = calcGain(entropyTwoAttributes(df,i,domains[i],domains[1]),[])
        if maxGain < gain
            maxGain = gain
            winner = i
        end
    end

    root = ID3(winner,ID3[])
    for featureVal in domains[winner]
        push!(root.children,createID3Recursive(df[df[:,winner] .== featureVal],domains[Not(winner)],setThreshold))
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
           [""],
           ["True","False"]]

permutecols!(df, [:Column5, :Column1,:Column2,:Column3,:Column4])


entropyTwoAttributes(df,2,domains[2],domains[1])
