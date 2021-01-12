using Distributions
using Printf

mutable struct Neuron
    value::Float64
    Δ::Float64
    parents::Array{Neuron,1}
    children::Array{Neuron,1}
    inputWeights::Array{Float64,1}
    Neuron()=new(Inf,Inf,[],[],[])
end

sigmoid(x)=1.0/(1.0+ℯ^(-x))

function activateNeuron!(neuron::Neuron)
    neuron.value = map(pair->pair[1].value * pair[2], zip(neuron.parents,neuron.inputWeights)) |>
           sum |> sigmoid
end

#the distribution includes the bias nodes
function createNeuralNetwork(neuronDistribution; addBias=true)
    offset = 0
    if addBias
        neuronDistribution = map(x->x+1,neuronDistribution)
        offset = 1
    end

    layers = Array{Neuron,1}[[Neuron() for j in 1:neuronDistribution[i]] for i in 1:length(neuronDistribution)]

    for i in 1:length(layers)
        if i != length(layers)
            layers[i][length(layers[i])].value=-1
        end

        for j in 1:length(layers[i])
            layers[i][j].parents =  i==1 || j == length(layers[i]) ? [] : layers[i-1] #bias nodes have no parents
            layers[i][j].children = i==length(layers)  ? [] : layers[i+1][1:length(layers[i+1])-offset]
            layers[i][j].inputWeights = [rand(Uniform(-0.05,0.05)) for _ in 1:length(layers[i][j].parents)]
        end
    end
    pop!(layers[length(layers)]) #remove redudant bias node in last layer
    return layers
end

function backpropagate!(neuralNetwork,input,output; α=0.5)
    hiddenLayers = neuralNetwork[2:length(neuralNetwork)-1]
    outputLayer = last(neuralNetwork)

    sig_gradient(x) = x * (1 - x) # the argmuent is the value of the neuron, which already is the value pf the sigmoid

    for i in 1:length(outputLayer)
        outputLayer[i].Δ = (output[i] - outputLayer[i].value) * sig_gradient(outputLayer[i].value)
        for j in 1:length(outputLayer[i].parents)
            outputLayer[i].inputWeights[j] += α * outputLayer[i].Δ * outputLayer[i].parents[j].value
        end
    end

    for i in reverse(1:length(hiddenLayers))
        for j in 1:length(hiddenLayers[i])
            println("--------------")
            lol = map(child->child.Δ * child.inputWeights[j], hiddenLayers[i][j].children)
            println(lol)
            println("---------------")
            temp = reduce(sum,lol)
            hiddenLayers[i][j].Δ = sig_gradient(hiddenLayers[i][j].value) * temp
            for k in 1:length(hiddenLayers[i][j].parents)
                hiddenLayers[i][j].inputWeights[k] += α * hiddenLayers[i][j].parents[k].value * hiddenLayers[i][j].Δ
            end
        end
    end
end

function forwardPropagate!(neuralNetwork,input)
    for j in 1:length(neuralNetwork[1])-1
        neuralNetwork[1][j].value=input[j]
    end

    for j in 2:length(neuralNetwork)
        for k in 1:length(neuralNetwork[j])
            activateNeuron!(neuralNetwork[j][k])
        end
    end
end

function train!(neuralNetwork,inputs,outputs; max_iters=50000, errThreshold=0.01)
    for i in 1:max_iters
        for z in 1:length(inputs)
            forwardPropagate!(neuralNetwork,inputs[z])
            backpropagate!(neuralNetwork,inputs[z],[outputs[z]])
        end
    end
end

networkInput=[[0,0],[1,0],[0,1],[1,1]]
networkOutput=[0,1,1,0]
neuralNetwork=createNeuralNetwork([2,4,2,1])

#=for i in 1:length(neuralNetwork)
    for j in 1:length(neuralNetwork[i])
        @printf("Neuron on layer %d and position %d has %d children and %d parents\n",i,j,length(neuralNetwork[i][j].children),length(neuralNetwork[i][j].parents))
    end
end=#

train!(neuralNetwork,networkInput,networkOutput)
#=
for tst in networkInput
    forwardPropagate!(neuralNetwork,tst)
    @printf("Input %s for XOR gate has result %.6f\n",tst,last(neuralNetwork)[1].value)
end
=#
