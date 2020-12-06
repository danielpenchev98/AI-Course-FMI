using DataFrames
using DataFramesMeta
using CSV
using Printf
using Plots
using Distributions
using Random
using InvertedIndices

function laplaceSmoothing(table::Array{Array{Float64,1},1}; α = 0)
    for i = 1:size(table, 1)
        all = 0
        for j = 1:size(table[i], 1)
            all += table[i][j]
        end

        for j = 1:size(table[i], 1)
            table[i][j] = (table[i][j] + α) / (all + α * size(table[i], 1))
        end
    end
    return table
end

#Create real condifitonal Tables
function getClassCountTables(
    df::DataFrame,
    class::Int64,
    domains::Array{Array{Int64,1},1},
)
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


function getProbabilityTables(df::DataFrame, domains::Array{Array{Int64,1},1})
    condPropTables = Array{Array{Float64,1},1}[]
    classPropTable = Float64[]

    for i = 1:size(domains[1], 1)
        classCount, condTables = getClassCountTables(df, i, domains)
        push!(classPropTable, classCount)
        push!(condPropTables, laplaceSmoothing(condTables))
    end

    return laplaceSmoothing(Array{Float64,1}[classPropTable])[1], condPropTables
end

function pickValue(valueProbabilities::Array{Float64,1}, domain::Array{Int64,1})
    result = rand(Multinomial(1, valueProbabilities))

    winner = -1
    for i = 1:size(result, 1)
        if result[i] == 1
            winner = i
            break
        end
    end

    return domain[winner]
end

function fillMissingValues(df::DataFrame, domains::Array{Array{Int64,1},1})
    subsets = DataFrame[]
    for c = 1:size(domains[1], 1)
        subset::DataFrame = @where(df, :x1 .== domains[1][c])

        classCount, condTables = getClassCountTables(subset, c, domains)
        for i = 1:size(condTables)[1]
            all::Float64 = foldr(+, condTables[i]; init = 0.0)
            ### all cannot be zero, because all columns with only missing df were removed

            for j = 1:size(condTables[i], 1)
                condTables[i][j] /= all
            end
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

#TODO better name
function getRandomPermOfSamples(df::DataFrame, n::Int64)
    df_size::Int64 = size(df, 1)
    seed::UInt64 = rand(UInt64)
    randvec = randperm!(MersenneTwister(seed), Vector{Int64}(undef, df_size))
    return randvec, convert(UInt64, round(df_size / n))
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
        classIdx =
            classify(validateSet[i, Not(1)], classPropTable, condPropTables)
        if classIdx == validateSet[i, 1]
            success += 1
        end
    end
    return (1.0 * success) / size(validateSet, 1)
end

function mapCategorialToIndex(df::DataFrame, domains::Array{Array{String,1},1}) #domains::Array{Array{Int8,1},1})
    newDf::DataFrame = DataFrame([Int8[] for i = 1:size(df, 2)])
    for i = 1:size(df, 1)
        push!(newDf, [0 for j = 1:size(df, 2)])
        for j = 1:size(df, 2)
            for k = 1:size(domains[j], 1)
                if df[i, j] == domains[j][k]
                    newDf[i, j] = k
                    break
                end
            end
        end
    end

    newDomains = Array{Int64,1}[]
    for i = 1:size(df, 2)
        push!(newDomains, [j for j = 1:size(domains[i], 1)])
    end
    return newDf, newDomains
end

#TODO Stratified K-fold
function naiveBayesClassification(df::DataFrame, domains::Array{Array{Int64,1},1}; k = 10)
    kFoldPlan::Array{Int64,1}, sampleSize::Int64 = getRandomPermOfSamples(df, 10)
    meanAccuracy::Float64 = 0

    for i = 1:k
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

df = CSV.read("/Users/i515142/Downloads/house-votes-84.data")

toDelete = []
for i = 1:size(df)[1]
    unknowns = 0

    for column in names(df)
        if df[i, column] == "?"
            unknowns = unknowns + 1
        end
    end
    if unknowns > 1.0 * size(df, 2) / 2
        push!(toDelete, i)
    end
end

delete!(df, toDelete)

######### REMOVE rows with > 50% missing df and columns with > 50 % missing values

domains = [["democrat", "republican"]]
for i = 2:size(df, 2)
    push!(domains, ["y", "n"])
end

df, domains = mapCategorialToIndex(df, domains)
df = fillMissingValues(df, domains)
df = df[shuffle(1:size(df, 1)), :];

naiveBayesClassification(df, domains)
