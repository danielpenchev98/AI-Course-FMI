using DataFrames
using DataFramesMeta
using CSV
using Printf
using Plots
using Distributions
using Random
using InvertedIndices

struct ID3Node
    feature::Int64
    children::Array{ID3Node,1}
    dominatingClass::String
    isLeaf::Bool
end

#TODO could only pass only the column with the classes
function entropyOneAttribute(df::DataFrame,classDomain::Array{String,1}; α=0.005)
    allCounts = nrow(df)
    propbabilities = map(class -> (count(df[:,1] .== class) + α) /(allCounts + α * length(classDomain[1])) ,classDomain)
    return - foldr((prob, res) -> res + prob * log2(prob) ,propbabilities,init=0.0)
end

function entropyTwoAttributes(df::DataFrame,feature::Int64,featureDomain::Array{String,1},classDomain::Array{String,1}; α = 0.005)
    allCounts, ∑ = nrow(df), 0.0
    for featureValue in featureDomain
        entropy = entropyOneAttribute( df[df[:,feature] .== featureValue,:],classDomain)
        propOfFeature = (count(df[:,feature] .== featureValue) + α) / (allCounts + α * length(featureDomain))
        ∑ += entropy * propOfFeature
    end
    return ∑
end

function getDominatingClass(df::DataFrame, classDomain::Array{String,1}, parentDominatingClass)
    winners = []
    dominatingCount = 0
    for class in classDomain
        cnt = count(df[:,1] .== class)
        if dominatingCount < cnt
            dominatingCount, winners = cnt, [class]
        elseif dominatingCount == cnt
            push!(winners,class)
        end
    end

    if length(winners) == 1
        return winners[1]
    elseif parentDominatingClass != "Sentinel" && parentDominatingClass in winners
        return parentDominatingClass
    end

    return winners[(rand(Int) % length(winners))+1]
end

function calcGain(df::DataFrame,featureIdx::Int64,domains::Array{Array{String,1},1},parentEntropy::Float64)
    childEntropy = entropyTwoAttributes(df,featureIdx,domains[featureIdx],domains[1])
    return parentEntropy - childEntropy
end

function getFeatureWithBestGain(df::DataFrame, domains::Array{Array{String,1},1},remainingFeatures::Array{Int64,1})
    winner, maxGain = -Inf, -Inf
    currEntropy = entropyOneAttribute(df,domains[1])
    for i in remainingFeatures
        currGain = calcGain(df,i,domains,currEntropy)
        if maxGain < currGain
            winner, maxGain = i, currGain
        end
    end
    return winner
end

function createID3Node(df::DataFrame,domains::Array{Array{String,1},1};setThreshold=14)
    if nrow(df) == 0
        throw(ArgumentError("The dataset is empty"))
    elseif size(df,2) != size(domains,1)
        throw(ArgumentError("Number of features doesnt match the number of the domains"))
    end

    remainingFeatures = [i for i in 2:size(df,2)]
    return createID3NodeRecursive(df,domains,remainingFeatures,setThreshold,"Sentinel")
end

function isHomogenous(df::DataFrame)
    return length(unique(df[:,1])) == 1
end

function createID3NodeRecursive(df::DataFrame,domains::Array{Array{String,1},1}, remainingFeatures::Array{Int64,1}, setThreshold::Int64, parentDominatingClass::String)
    if nrow(df) == 0
        return ID3Node(-1,ID3Node[],parentDominatingClass,true)
    end

    dominatingClass = getDominatingClass(df,domains[1],parentDominatingClass)

    if nrow(df) <= setThreshold || length(remainingFeatures) == 0 || isHomogenous(df)
        return ID3Node(-1,ID3Node[],dominatingClass,true)
    end

    winner = getFeatureWithBestGain(df,domains,remainingFeatures)
    remainingFeatures = filter(e-> e ≠ winner,remainingFeatures)

    children = ID3Node[]
    for featureVal in domains[winner]
        filteredSet = df[df[:,winner] .== featureVal,:]
        child = createID3NodeRecursive(filteredSet,domains,remainingFeatures,setThreshold,dominatingClass)
        push!(children, child)
    end
    return ID3Node(winner,children,dominatingClass,false)
end

function pickValue(valueProbabilities::Array{Float64,1}, domain::Any)
    result = rand(Multinomial(1, valueProbabilities))
    return domain[findfirst(x -> x == 1, result)]
end

#TODO think of a true reason why the some of the demain values of this feature have 0% chance - maybe laplace smoothing + normalization
function fillValues!(df::DataFrame, domains::Array{Array{String,1},1})
    for col in 1:ncol(df)
        cnts = map(val -> count(df[:,col] .== val), domains[col])
        distribution = 1.0 * cnts ./ sum(cnts)
        missingValueRows = foldr((row, res) -> ismissing(df[row,col]) ? push!(res,row) : res, 1:nrow(df), init=[])
        for row in missingValueRows
            df[row,col] = pickValue(distribution,domains[col])
        end
    end
end

function classify(entity::DataFrameRow,tree::ID3Node, domains::Array{Array{String,1},1})
    currNode = tree
    while !currNode.isLeaf
        entityFeatureVal = entity[currNode.feature-1] #entity has no class column, which is the first column
        childIndx = findfirst(x -> x == entityFeatureVal, domains[currNode.feature])
        if isnothing(childIndx)
            break
        end
        currNode = currNode.children[childIndx]
    end
    return currNode.dominatingClass
end

function validateModel(validateSet::DataFrame,tree::ID3Node, domains::Array{Array{String,1},1})
    success::Int64 = 0
    for i = 1:size(validateSet, 1)
        classIdx = classify(validateSet[i, Not(1)],tree,domains)
        if classIdx == validateSet[i, 1]
            success += 1
        end
    end
    return (1.0 * success) / size(validateSet, 1)
end

function stratifiedKFold(df::DataFrame,classDomain::Array{String,1}, folds::Int64)
    indeces = [i for i in 1:size(df,1)]
    indecesPerClass = map((class) -> filter(x->df[x,1]==class,indeces) ,classDomain)
    distributions = map((classIndeces) -> size(classIndeces,1) / size(indeces,1) ,indecesPerClass)

    plan = []
    foldSize = convert(Int64, floor(size(df,1)/folds))
    for i in 1:folds
        for j in 1:size(classDomain,1)
            chunkSize = convert(Int64 ,floor(distributions[j] * foldSize +  ((i+j)%2==0 ? 1 : 0)))
            append!(plan,indecesPerClass[j][1:chunkSize])
            deleteat!(indecesPerClass[j],1:chunkSize)
        end
    end

    return plan, foldSize
end

function DecisionTreeClassification(df::DataFrame, domains::Array{Array{String,1},1}; kfold = 10, setMinPopulation = 14)
    kFoldPlan::Array{Int64,1}, sampleSize::Int64 = stratifiedKFold(df,domains[1],kfold)
    meanAccuracy::Float64 = 0

    for i in 1:kfold
        trainDataSetIndeces = kFoldPlan[Not((i-1)*sampleSize+1:i*sampleSize)]
        tree = createID3Node(df[trainDataSetIndeces,:],domains, setThreshold = setMinPopulation)

        accuracy::Float64 = validateModel( df[Not(trainDataSetIndeces),:],tree,domains)
        meanAccuracy += accuracy / kfold
    end

    return meanAccuracy
end

#=
df = CSV.read("./data/breast-cancer.data",header=false,types=[String for i in 1:10])

const missingValuesThreshold = 1.0 * size(df, 2) / 2
const missingValueSign = "?"

domains = [unique(filter(x->x!=missingValueSign,df[:,i])) for i in 1:ncol(df)]
fillValues!(df,domains)

bestPopulationThreshold = 1
bestMeanAccuracy = 0.0

for i in 1:20
    df = df[shuffle(1:size(df, 1)), :];
    meanAcc = DecisionTreeClassification(df,domains,setMinPopulation = i)
    if bestMeanAccuracy < meanAcc
        bestMeanAccuracy = meanAcc
        bestPopulationThreshold = i
    end
end

@printf("Mean accuracy %.6f with population threshold %d\n",bestMeanAccuracy,bestPopulationThreshold)
=#
