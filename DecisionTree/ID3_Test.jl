using Test
include("ID3.jl")
data = CSV.read("./data/golf.csv",header=false,types=[String for i in 1:5])
permutecols!(data,[:Column5,:Column1,:Column2,:Column3,:Column4])
domains = [unique(data[:,i]) for i in 1:size(data,2)]

ϵ = 0.03
weatherFeatureIdx = 2

@testset "ID3 algorithm" begin
    @testset "entropy calculation" begin
        @testset "single attribute" begin
            #could vary because of the laplace smoothing
            @test abs(entropyOneAttribute(data,domains[1]) - 0.94) ≤ ϵ
        end
        @testset "two attributes" begin
            #could vary because of the laplace smoothing
            @test abs(entropyTwoAttributes(data,weatherFeatureIdx,domains[weatherFeatureIdx],domains[1]) - 0.693) ≤ ϵ
        end
    end
    @testset "dominating class exploration" begin
        @testset "returns domnating class" begin
            @test getDominatingClass(data,domains[1],"Some parent") == "Yes"
        end
        @testset "equal probability and parent not root" begin
            #when equal probability, take into account parent
            @test getDominatingClass(data[data[:,3] .== "Hot",:],domains[1],"No") == "No"
        end
    end
end;
