using DataFrames
using DataFramesMeta
using CSV
using Printf
using Plots
using Distributions
using Random
using InvertedIndices

#Create real condifitonal Tables
function getClassCountTables(df::DataFrame, class::Int64, domains::Array{Array{Int64,1},1})
    condTable = [[0.0 for j = 1:size(domains[i], 1)] for i = 2:size(df, 2)]
    classCount::Float64 = 0.0

    for i = 1:size(df, 1)
        if df[i, 1] != class
            continue
        end
        classCount += 1

        for j = 1:size(condTable, 1)
            for k = 1:size(condTable[j], 1)
                #j+1 because we start from the second column -> the first feature column
                if df[i, j+1] == domains[j+1][k]
                    condTable[j][k] += 1.0
                end
            end
        end
    end
    return classCount, condTable
end

function pickValue(valueProbabilities::Array{Float64,1}, domain::Array{Int64,1})
    result = rand(Multinomial(1, valueProbabilities))
    return domain[findfirst(x -> x == 1, result)]
end

function fillMissingValues(df::DataFrame, domains::Array{Array{Int64,1},1}, missingValueMark::Int64)
    subsets = DataFrame[]
    for c = 1:size(domains[1], 1)
        subset::DataFrame = @where(df, :x1 .== domains[1][c])

        classCount, condTables = getClassCountTables(subset, c, domains)
        for i = 1:size(condTables)[1]
            all::Float64 = foldr(+, condTables[i]; init = 0.0)
            condTables[i] = map((val)->val/all,condTables[i])
        end

        for i = 1:size(subset)[1]
            for j = 2:size(subset, 2)
                #0 is the missing value, the min value from the domain is 1
                if subset[i, j] < domains[j][1]
                    subset[i, j] = pickValue(condTables[j-1], domains[j])
               end

            end
        end
        push!(subsets, subset)
    end

    while size(subsets, 1) > 1

        append!(subsets[1], pop!(subsets))
    end

    return pop!(subsets)
end

function laplaceSmoothing(table::Array{Array{Float64,1},1}; α = 0.1)
    for i = 1:size(table, 1)
        all::Float64 = sum(table[i])
        for j = 1:size(table[i], 1)
            table[i][j] = (table[i][j] + α) / (all + α * size(table[i], 1))
        end
    end
    return table
end

function getProbabilityTables(df::DataFrame, domains::Array{Array{Int64,1},1})
    condPropTables = Array{Array{Float64,1},1}[] # P(feature | class)
    classPropTable = Float64[] # P(class)

    for i = 1:size(domains[1], 1)
        classCount, condTables = getClassCountTables(df, i, domains)
        push!(classPropTable, classCount)
        push!(condPropTables, laplaceSmoothing(condTables))
    end

    return laplaceSmoothing(Array{Float64,1}[classPropTable])[1], condPropTables
end


function classify(entity::DataFrameRow,classPropTable::Array{Float64,1},condPropTables::Array{Array{Array{Float64,1},1},1})
    bestVal::Float64 = -Inf
    bestClassIdx::Int64 = 1

    for i = 1:size(classPropTable, 1)
        result::Float64 = foldr(
            (idx, oldVal) -> oldVal + log(condPropTables[i][idx][entity[idx]]),
            1:size(entity, 1),
            init = log(classPropTable[i]),
        )

        if bestVal < result || bestVal == -Inf
            bestVal = result
            bestClassIdx = i
        end
    end
    return bestClassIdx
end

function validateModel(validateSet::DataFrame,classPropTable::Array{Float64,1},condPropTables::Array{Array{Array{Float64,1},1},1})
    success::Int64 = 0
    for i = 1:size(validateSet, 1)
        classIdx = classify(validateSet[i, Not(1)], classPropTable, condPropTables)
        if classIdx == validateSet[i, 1]
            success += 1
        end
    end
    return (1.0 * success) / size(validateSet, 1)
end

function mapCategorialToIndex(df::DataFrame, mapping::Dict{String,Int64}, domainSizes::Array{Int64,1})
    #For some unexplainable reason Dataframe takes columns as arguments
    newDf = DataFrame(map(
        (col) -> map((row) -> mapping[df[row, col]], 1:size(df, 1)),
        1:size(df, 2)
    ))

    domains = Array{Int64,1}[[j for j = 1:domainSizes[i]] for i in 1:size(df,2)]
    return newDf, domains
end

#fold size should be even number
function stratifiedKFold(df::DataFrame,classDomain::Array{Int64,1}, folds::Int64)
    indeces = [i for i in 1:size(df,1)]
    indecesPerClass = map((class) -> filter(x->df[x,1]==class,indeces) ,classDomain)
    distributions = map((classIndeces) -> size(classIndeces,1) / size(indeces,1) ,indecesPerClass)

    plan = []
    foldSize = convert(Int64, round(size(df,1)/folds))
    for i in 1:folds
        for j in 1:size(classDomain,1)
            chunkSize = convert(Int64 ,floor(distributions[j] * foldSize + ((i+j)%2==0 ? 1 : 0)))
            append!(plan,indecesPerClass[j][1:chunkSize])
            deleteat!(indecesPerClass[j],1:chunkSize)
        end
    end

    return plan, foldSize
end

function naiveBayesClassification(df::DataFrame, domains::Array{Array{Int64,1},1}; k = 10)
    kFoldPlan::Array{Int64,1}, sampleSize::Int64 = stratifiedKFold(df,domains[1],10)
    meanAccuracy::Float64 = 0

    for i in 1:k
        testDataSetIndeces = kFoldPlan[Not((i-1)*sampleSize+1:i*sampleSize)]

        classPropTable, condProbTables =
            getProbabilityTables(df[testDataSetIndeces, :], domains)

        accuracy::Float64 = validateModel(
            df[Not(testDataSetIndeces), :],
            classPropTable,
            condProbTables,
        )
        @printf("Training %d with accurancy %.6f\n", i, accuracy)
        meanAccuracy += accuracy / k
    end
    @printf("Mean accurancy %.6f\n", meanAccuracy)
end

df = CSV.read("./data/house-votes-84.data",header=false)

const missingValuesThreshold = 1.0 * size(df, 2) / 2
const missingValueSign = "?"

rowsToDelete = filter(rowIdx -> foldr(
            (colIdx, y) -> y + (df[rowIdx, colIdx] == missingValueSign ? 1 : 0),
            1:size(df, 2),
            init = 0,
    ) > missingValuesThreshold,
    1:size(df, 1),
)

deleterows!(df,rowsToDelete)

domainSizes = Int64[ 2 for i in 1:size(df,2)]
mapping = Dict("democrat" => 1, "republican" => 2, "?" => 0, "y" => 1, "n" => 2)
df, domains = mapCategorialToIndex(df, mapping, domainSizes)

const missingValueMark = 0
df = fillMissingValues(df, domains, missingValueMark)

df = df[shuffle(1:size(df, 1)), :];
@time naiveBayesClassification(df, domains)
