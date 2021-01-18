include("ID3.jl")

struct RandomForest
    trees::Array{ID3Node,1}
end

function createBoostingDataset(df::DataFrame,classDomain::Array{String,1})
    fullSetIndeces = [i for i in 1:size(df,1)]
    trainingSetIndeces = sample(fullSetIndeces,length(fullSetIndeces),replace=true)
    validationSetIndeces = setdiff(fullSetIndeces,trainingSetIndeces)
    #validationSetIndeces = filter(id -> df[id,1]=="no-recurrence-events",1:size(df,1))
    return trainingSetIndeces, validationSetIndeces
end

function createID3Tree2(df::DataFrame,domains::Array{Array{String,1},1}; depthThreshold=3, setThreshold=12)
    if nrow(df) == 0
        throw(ArgumentError("The dataset is empty"))
    elseif size(df,2) != size(domains,1)
        throw(ArgumentError("Number of features doesnt match the number of the domains"))
    end

    remainingFeatures = [i for i in 2:size(df,2)] #the first column is the class column
    return createID3TreeRecursive2(df,domains,remainingFeatures,setThreshold,"Sentinel")
end

function isHomogenous(df::DataFrame)
    return length(unique(df[:,1])) == 1
end

function createID3TreeRecursive2(df::DataFrame,domains::Array{Array{String,1},1}, remainingFeatures::Array{Int64,1},setThreshold::Int64,parentDominatingClass::String)
    if nrow(df) == 0
        return ID3Node(-1,ID3Node[],parentDominatingClass,true)
    end

    dominatingClass = getDominatingClass(df,domains[1],parentDominatingClass)

    if nrow(df) <= setThreshold || length(remainingFeatures) == 5 || isHomogenous(df)
        return ID3Node(-1,ID3Node[],dominatingClass,true)
    end

    comp::Array{Int64,1} = sample(remainingFeatures,convert(Int64,ceil(sqrt(length(remainingFeatures)))),replace=false)
    winner = getFeatureWithBestGain(df,domains,comp)
    remainingFeatures = filter(e-> e ≠ winner,remainingFeatures)

    children = ID3Node[]
    for featureVal in domains[winner]
        filteredSet = df[df[:,winner] .== featureVal,:]
        child = createID3TreeRecursive2(filteredSet,domains,remainingFeatures,setThreshold,dominatingClass)
        push!(children, child)
    end
    return ID3Node(winner,children,dominatingClass,false)
end

#
function createRandomForest(df::DataFrame,domains::Array{Array{String,1},1};treeNumber=10)
    trees = ID3Node[]
    outOfBagSets = []
    for i in 1:treeNumber
        trainDataSetIndeces, validationSetIndeces = createBoostingDataset(df,domains[1])
        tree = createID3Tree2(df[trainDataSetIndeces,:],domains)
        push!(trees,tree)
        push!(outOfBagSets,validationSetIndeces)
    end
    return trees, outOfBagSets
end

function validateRandomForest(df::DataFrame,trees::Array{ID3Node,1},outOfBagSets::Array{Any,1})
    tests=[]
    for recordId in 1:size(df,1)
        votes = String[]
        for i in 1:size(outOfBagSets,1)
            if recordId ∉ outOfBagSets[i]
                continue
            end
            push!(votes,classify(df[recordId,:],trees[i],domains))
        end
        if length(votes) == 0
            continue
        end

        countTuples = map(class -> (class,count(vote->vote==class,votes)),domains[1])
        winners = foldr((x,res) -> res==() || x[2]>res[2] ? ([x[1]],x[2]) : x[2]==res[2] ? (push!(res[1],x[1]),res[2]) : res, countTuples, init=([],0))
        if length(winners[1]) > 1
            push!(tests,false)
        else
            #@printf("Real %s and is classified as %s\n",df[recordId,1],winners[1][1])
            if winners[1][1] == "recurrence-events"
                println("YAYAYA")
            end
            push!(tests,df[recordId,1] == winners[1][1] )
        end
    end
    return 1.0 * count(x->x==true,tests) / length(tests)
end



df = DataFrame(CSV.read("./data/breast-cancer.data",header=false,types=[String for i in 1:10]))

const missingValuesThreshold = 1.0 * size(df, 2) / 2
const missingValueSign = "?"

domains = [unique(filter(x->x!=missingValueSign,df[:,i])) for i in 1:ncol(df)]
fillValues!(df,domains)
df = unique(df)

df = df[shuffle(1:size(df, 1)), :];

trees, outOfBagSets = createRandomForest(df,domains)
accuracy = validateRandomForest(df,trees,outOfBagSets)
println(accuracy)
println(accuracy)
#println(accuracy)
