using DataFrames
using DataFramesMeta
using CSV
using Printf
using Plots
using Distributions
using Random
using InvertedIndices

function laplaceSmoothing(table; α = 1)
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
function getClassCountTables(df::DataFrame, class, domains)
    condTable = [[0.0 for j = 1:size(domains[i], 1)] for i = 2:size(df, 2)]
    classCount = 0.0

    for i = 1:size(df, 1)
        if df[i, 1] != class
            continue
        end
        classCount += 1
        for j = 1:size(condTable, 1)
            for k = 1:size(condTable[j], 1)
                if df[i, j+1] == domains[j+1][k]
                    condTable[j][k] += 1.0
                end
            end
        end
    end
    return classCount, condTable
end


function getProbabilityTables(df::DataFrame, domains)
    condPropTables = []
    classPropTable = []

    for i = 1:size(domains[1], 1)
        classCount, condTables = getClassCountTables(df, i, domains)
        push!(classPropTable, classCount)
        push!(condPropTables, laplaceSmoothing(condTables))
    end

    return laplaceSmoothing([classPropTable])[1], condPropTables
end

function pickValue(votes, domain)
    result = rand(Multinomial(1, votes))
    winner = -1
    for i = 1:size(result, 1)
        if result[i] == 1
            winner = i
            break
        end
    end

    return domain[winner]
end

function fillMissingValues(df::DataFrame, domains)
    subsets = []
    for c = 1:size(domains[1], 1)
        subset = @where(df, :x1 .== domains[1][c])

        classCount, votes = getClassCountTables(subset, c, domains)
        for i = 1:size(votes)[1]
            all = 0
            for j = 1:size(domains[i], 1)
                all += votes[i][j]
            end

            ### all cannot be zero, because all columns with only missing df were removed

            for j = 1:size(votes[i], 1)
                votes[i][j] /= all
            end
        end

        for i = 1:size(subset)[1]
            for j = 2:size(subset, 2)
                #0 is the missing value, the min value from the domain is 1
                if subset[i, j] < domains[j][1]
                    subset[i, j] = pickValue(votes[j-1], domains[j])
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
function splitRandomlyDataset(df::DataFrame, n::Int64)
    df_size = size(df, 1)
    seed = rand(UInt64)
    randvec = randperm!(MersenneTwister(seed), Vector{Int64}(undef, df_size))
    return randvec, convert(UInt64, round(df_size / n))
end

function classify(entity::DataFrameRow, classPropTable, condPropTables)
    bestVal = -Inf
    bestClassIdx = 1

    for i = 1:size(classPropTable, 1)
        result = log(classPropTable[i])
        temp = 0
        for j = 1:size(entity, 1)
            result += log(condPropTables[i][j][entity[j]])
        end

        if bestVal < result || bestVal == -Inf
            bestVal = result
            bestClassIdx = i
        end
    end

    return bestClassIdx
end

function validateModel(validateSet::DataFrame, classPropTable, condPropTables)
    success = 0
    for i = 1:size(validateSet, 1)
        classIdx =
            classify(validateSet[i, Not(1)], classPropTable, condPropTables)
        if classIdx == validateSet[i, 1]
            success += 1
        end
    end
    return success / size(validateSet, 1)
end

function mapColumnsToIndex(df, domains)
    newDf = DataFrame([Int8[] for i = 1:size(df, 2)])
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

    newDomains = []
    for i = 1:size(df, 2)
        push!(newDomains, [j for j = 1:size(domains[i], 1)])
    end
    return newDf, newDomains
end

#TODO Stratified K-fold
function naiveBayesClassification(df::DataFrame, domains; k = 10)
    modifiedDf = mapColumnsToIndex(df, domains)
    kFoldPlan, setSize = splitRandomlyDataset(df, k)
    meanAccuracy = 0
    for i = 1:k
        testDataSetIndeces = kFoldPlan[Not((i-1)*setSize+1:i*setSize)]
        classPropTable, condProbTables =
            getProbabilityTables(df[testDataSetIndeces, :], domains)
        accuracy = validateModel(
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

df, domains = mapColumnsToIndex(df, domains)
df = fillMissingValues(df, domains)
df = df[shuffle(1:size(df, 1)), :];
naiveBayesClassification(df, domains)
