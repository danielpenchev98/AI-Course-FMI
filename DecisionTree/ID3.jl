using DataFrames
using DataFramesMeta
using CSV
using Printf
using Plots
using Distributions
using Random
using InvertedIndices

#TODO at confusion matrix for accuracy evaluation

mutable struct ID3Node
    feature::Int64 #from 1 to the number of columns, none determined if -1
    children::Array{ID3Node,1} #sub decision trees
    dominatingClass::String #the dominating class currently
    isLeaf::Bool #if terminal node
 end

 mutable struct ConfusionMatrix
     truePositive::Int64
     falsePositive::Int64
     trueNegative::Int64
     falseNegative::Int64
 end

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
    winners, dominatingCount = [], 0
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
    else
        return winners[(rand(UInt64) % length(winners))+1]
    end

end

function calcGain(df::DataFrame,featureIdx::Int64,domains::Array{Array{String,1},1},parentEntropy::Float64)
    childEntropy = entropyTwoAttributes(df,featureIdx,domains[featureIdx],domains[1])
    return parentEntropy - childEntropy
end


function getFeatureWithBestGain(df::DataFrame, domains::Array{Array{String,1},1},competingFeatures::Array{Int64,1})
    winner, maxGain = -Inf, -Inf
    currEntropy = entropyOneAttribute(df,domains[1])
    for i in competingFeatures
        currGain = calcGain(df,i,domains,currEntropy)
        if maxGain < currGain
            winner, maxGain = i, currGain
        end
    end
    return winner
end


function createID3Tree(df::DataFrame,domains::Array{Array{String,1},1};setThreshold=14)
    if nrow(df) == 0 
        throw(ArgumentError("The dataset is empty"))
    elseif size(df,2) != size(domains,1)
        throw(ArgumentError("Number of features doesnt match the number of the domains"))
    end

    remainingFeatures = [i for i in 2:size(df,2)] #the first column is the class column
    return createID3TreeRecursive(df,domains,remainingFeatures,setThreshold,"Sentinel")
end

function isHomogenous(df::DataFrame)
    return length(unique(df[:,1])) == 1
end

function createID3TreeRecursive(df::DataFrame,domains::Array{Array{String,1},1}, remainingFeatures::Array{Int64,1}, setThreshold::Int64, parentDominatingClass::String)
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
        child = createID3TreeRecursive(filteredSet,domains,remainingFeatures,setThreshold,dominatingClass)
        push!(children, child)
    end
    return ID3Node(winner,children,dominatingClass,false)
end


function pickValue(valueProbabilities::Array{Float64,1}, domain::Any)
    result = rand(Multinomial(1, valueProbabilities))
    return domain[findfirst(x -> x == 1, result)]
end

function fillValues!(df::DataFrame, domains::Array{Array{String,1},1})
    for col in 1:ncol(df)
        cnts = map(val -> count(df[:,col] .== val), domains[col])
        distribution = 1.0 * cnts ./ sum(cnts)
        missingValueRows = foldr((row, res) -> df[row,col]=="?" ? push!(res,row) : res, 1:nrow(df), init=[])
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
    for test in eachrow(validateSet)
        classIdx = classify(test[Not(1)],tree,domains) #skip first column
        if classIdx == test[1]
            success += 1
        end
    end
    return (1.0 * success) / size(validateSet, 1)
end

#Only for binary classification
function validateModelConfusionMatrix(validateSet::DataFrame,tree::ID3Node, domains::Array{Array{String,1},1})
    confMatrix = ConfusionMatrix(0.0,0.0,0.0,0.0)
    for test in eachrow(validateSet)
        classIdx = classify(test[Not(1)],tree,domains) #skip first column
        if classIdx == 2
            if classIdx == test[1]
                confMatrix.truePositive += 1
            else
                confMatrix.falsePositive += 1
            end
        elseif classIdx == test[1]
            confMatrix.trueNegative += 1
        else
            confMatrix.falseNegative += 1
        end

    end
    return confMatrix
end

function stratifiedKFold(df::DataFrame,classDomain::Array{String,1}, folds::Int64)
    indeces = [i for i in 1:size(df,1)]
    indecesPerClass = map((class) -> filter(x->df[x,1]==class,indeces) ,classDomain)
    distributions = map((classIndeces) -> size(classIndeces,1) / size(indeces,1) ,indecesPerClass)

    plan = []
    foldSize = convert(Int64, floor(size(df,1)/folds))
    for i in 1:folds
        fold = []
        for j in 1:size(classDomain,1)
            chunkSize = min(length(indecesPerClass[j]),convert(Int64 ,ceil(distributions[j] * foldSize)))
            append!(fold,indecesPerClass[j][1:chunkSize])
            deleteat!(indecesPerClass[j],1:chunkSize)
        end
        push!(plan,fold)
    end

    return plan
end

function DecisionTreeClassification(df::DataFrame, domains::Array{Array{String,1},1}; kfold = 20, setMinPopulation = 14)
    kFoldPlan::Array{Array{Int64,1},1} = stratifiedKFold(df,domains[1],kfold)
    meanAccuracy::Float64 = 0

    for i in 1:kfold
        trainDataSetIndeces = reduce(vcat,kFoldPlan[Not(i)]) #merge all training folds into one
        tree = createID3Tree(df[trainDataSetIndeces,:],domains, setThreshold = setMinPopulation)

        accuracy::Float = validateModel(df[kFoldPlan[i],:],tree,domains)

        meanAccuracy += accuracy / kfold
    end

    return meanAccuracy
end

function calculateExpectedDistributions(parentDistribution::Array{Int64,1},childrenDistribution::Array{Array{Int64,1},1})::Array{Array{Float64,1},1}
    expectedProbabilities = map(childDistribution -> 1.0 * sum(childDistribution) / nrow(df),childrenDistribution) # p(first children) = # of elements in subset / # of elements in the set
    expectedDistributions = []
    for probability in expectedProbabilities
        expectedCount = map(distribution -> distribution * probability,parentDistribution)
        push!(expectedDistributions,expectedCount)
    end

    return expectedDistributions
end

function isPrunable(parentDistribution::Array{Int64,1}, childrenDistributions::Array{Array{Int64,1},1}, hypothesisThreshold::Float64)::Bool
    expectedDistributions = calculateExpectedDistributions(parentDistribution, childrenDistributions)

    K = 0.0
    for i in 1:length(childrenDistributions)
        for j in 1:length(childrenDistributions[i])
            K+= (childrenDistributions[i][j] - expectedDistributions[i][j])^2 / expectedDistributions[i][j]
        end
    end

    numberOfClasses, numberOfChildren = length(parentDistribution), length(childrenDistributions)
    degreeOfFreedom = (numberOfClasses - 1) * (numberOfChildren - 1)
    pValue = ccdf(Chisq(degreeOfFreedom),K)

    return pValue > hypothesisThreshold
end

function chiSquarePruning(root::ID3Node,df::DataFrame, domains::Array{Array{String,1},1}; threshold = 0.05)::Tuple{Array{Int64,1},Bool}
    currentDistribution::Array{Int64,1} = map(class -> count(df[:,1] .== class) ,domains[1]) # N(class A), N(class B) ...
    if root.isLeaf
        return (currentDistribution,true)
    end
    
    childrenDistributions::Array{Array{Int64,1},1} = []
    areChildrenLeafs = true
    for i in 1:length(root.children)
        featureVal = domains[root.feature][i]
        filteredSet = df[df[:,root.feature] .== featureVal,:] # Get subset of elements, whose value of the particular feature is ...
        childDistribution, isLeafs = chiSquarePruning(root.children[i],filteredSet,domains) # Recursively apply the function for the children first
        areChildrenLeafs &= isLeafs
        push!(childrenDistributions, childDistribution)
    end 

    if areChildrenLeafs && isPrunable(currentDistribution,childrenDistributions,threshold) #prune the children only when the condition is true and the children are themselves leafs
        root.children = ID3Node[]
        root.feature = -1                
        root.isLeaf = true
        return (currentDistribution, true)
    end

    return (currentDistribution,false)
end

#Binary classification
function DecisionTreeClassificationConfMatrix(df::DataFrame, domains::Array{Array{String,1},1}; kfold = 10, setMinPopulation = 14)
    kFoldPlan::Array{Array{Int64,1},1} = stratifiedKFold(df,domains[1],kfold)
    accumulativeConfMatrix = ConfusionMatrix(0.0,0.0,0.0,0.0)

    for i in 1:kfold
        trainDataSetIndeces = reduce(vcat,kFoldPlan[Not(i)]) #merge all training folds into one
        tree = createID3Tree(df[trainDataSetIndeces,:],domains, setThreshold = setMinPopulation)
        chiSquarePruning(tree,df[trainDataSetIndeces,:],domains)
        confMatrix = validateModelConfusionMatrix(df[kFoldPlan[i],:],tree,domains)

        accumulativeConfMatrix.truePositive += confMatrix.truePositive
        accumulativeConfMatrix.falsePositive += confMatrix.falsePositive
        accumulativeConfMatrix.trueNegative += confMatrix.trueNegative
        accumulativeConfMatrix.falseNegative += confMatrix.falseNegative
    end

    return accumulativeConfMatrix
end

function checkForFeatures(remainingFeatures::Array{Int64,1})
    featuresSets = []
    for i in 2:length(remainingFeatures)
        set = [i]
        for j in i:length(remainingFeatures)
            if j != i
                push!(set,j)
            end

            push!(featuresSets,copy(set))

            if j != i
                pop!(set)
            end
        end
    end
    return featuresSets
end


df = CSV.File("./data/breast-cancer.data", header=false,types=[String for i in 1:10]) |> DataFrame |> unique

println("WTF")
const missingValuesThreshold = 1.0 * size(df, 2) / 2
const missingValueSign = "?"

domains = [unique(filter(x->x!=missingValueSign,df[:,i])) for i in 1:ncol(df)]
fillValues!(df,domains)

df = df[shuffle(1:size(df, 1)), :];
confMatrix = DecisionTreeClassificationConfMatrix(df[:,Not([2,5])], domains[Not(2,5)],setMinPopulation = 12)
println(confMatrix)