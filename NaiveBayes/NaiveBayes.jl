using DataFrames
using DataFramesMeta
using CSV
using Printf
using Plots
using Distributions
using Random
using InvertedIndices

function laplaceSmoothing(table; α = 1.0)
    for i = 1:size(table,1)
        all = 0
        for j = 1:size(table[i],2)
            all += table[i][j]
        end

        for j = 1:size(table[i],2)
            table[i][j] = (table[i][j] + α) / (all + α * size(table[i],2))
        end
    end
    return table
end

function getCountTables(df::DataFrame, domains)
    table = [[0.0, 0.0] for i = 1:size(df,2)]

    for i = 1:size(df)[1]
        for j = 1:size(df,2)
            for k = 1:size(domains[j],1)
                if df[i, j] == domains[j][k]
                    table[j][k] += 1
                end
            end
        end
    end
    return table
end

function getConditionalProbabilityTables(df::DataFrame, domains)
    return laplaceSmoothing(getCountTables(df, domains))
end

function pickValue(votes, domain)
    result = rand(Multinomial(1, votes))
    winner = -1
    for i = 1:size(result,1)
        if result[i] == 1
            winner = i
            break
        end
    end

    return domain[winner]
end

function fillMissingValues(df::DataFrame, domains)
    subsets = []
    for c in 1:size(domains[1],1)
        subset = @where(df, :col1 .== domains[1][c])

        votes = getCountTables(subset,domains)
        for i = 1:size(votes)[1]
            all = 0
            for j = 1:size(domains[i],1)
                all += votes[i][j]
            end

            ### all cannot be zero, because all columns with only missing df were removed

            for j = 1:size(votes[i],1)
                votes[i][j] /= all
            end
        end

        for i = 1:size(subset)[1]
            for j = 2:size(subset,2)
                if subset[i, j] == "?"
                    subset[i, j] = pickValue(votes[j], domains[j])
                end
            end
        end
        push!(subsets,subset)
    end

    while size(subsets,1) > 1
        append!(subsets[1],pop!(subsets))
    end

    return pop!(subsets)
end

function splitRandomlyDataset(df::DataFrame, n::Int64)
    #dfs = []
    df_size = size(df,1)

    seed = rand(UInt64)
    randvec = randperm!(MersenneTwister(seed),Vector{Int64}(undef,df_size))

    #iter = 1
    #sample_size = convert(UInt64,round(df_size / n))
    #for i in 1:n
    #    sample = df[iter:min(iter+sample_size-1,size(df,1)),:]
    #    push!(dfs,sample)
    #    iter += sample_size
    #end
    #return dfs
    return randvec
end

function classify(entity::DataFrame,condPropTables)
    bestVal = 0
    bestClassIdx = 1
    for i in 1:size(condPropTables[1],1)
        result = log(condPropTables[1][i])
        temp = 0
        for j in 2:size(entity,2)
            result += log(condPropTables[j][i])
        end

        if bestVal < result
            bestVal = result
            bestClassIdx = i
        end
    end

    return bestClassIdx
end

function validateModel(validateSet::DataFrame, condPropTables )
    success = 0
    for i in 1:size(validateSet,1)
        classIdx = classify(validateSet[i,Not(1)],condPropTables)
        if classNames[classIdx] == validateSet[i,1]
            success+=1
        end
    end
    return success/size(validateSet,1)
end

#TODO Stratified K-fold
function naiveBayesClassification(df::DataFrame,domains)
    k=10
    kFoldPlan = splitRandomlyDataset(df,k)
    meanAccuracy = 0
    for i in 1:k
        testDataSetIndeces = kFoldPlan[Not((i-1)*k+1:i*k)]
        condProbTables = getConditionalProbabilityTables(df[testDataSetIndeces,:],domains)
        println(condProbTables)
        accuracy = validateModel(df[i,:],condProbTables)
        @printf("Training %d with accurancy %.6f\n",i,accuracy)
        meanAccuracy += accuracy/k
    end

    @printf("Mean accurancy %.6f\n",meanAccuracy)
end


df = CSV.read("/Users/i515142/Downloads/house-votes-84.data")
names!(df, [Symbol("col$i") for i = 1:17])


for column in names(df)
    unknown = 0
    for value in df[!, column]
        if value == "?"
            unknown = unknown + 1
        end
    end
    @printf(
        "Column name %s with ration of unknows/known = %d/%d\n",
        column,
        unknown,
        size(df)[1]
    )
end


maxUnknowns = 0
worstRow = 1


toDelete = []
missingDistr = []
for i = 1:size(df)[1]
    unknowns = 0
    for column in names(df)
        if df[i, column] == "?"
            unknowns = unknowns + 1
        end
    end
    push!(missingDistr, unknowns)
    if maxUnknowns < unknowns
        maxUnknowns = unknowns
        worstRow = i
    end

    if unknowns > 1.0 * size(df)[2] / 2
        push!(toDelete, i)
    end
end

delete!(df, toDelete)

#println(missingDistr)
#h = histogram(missingDistr)
#plot!(h)

######### REMOVE rows with > 50% missing df and columns with > 50 % missing values

domains = [["democrat", "republican"]]
for i = 2:size(df)[2]
    push!(domains, ["y", "n"])
end

#datasets = splitRandomlyDataset(df,10)
#for i in 1:10
#    @printf("-----------Dataset %d-----------\n",i)
#    println(datasets[i])
#    println()
#end


df = fillMissingValues(df,domains)
naiveBayesClassification(df,domains)
