using Test
using Random
using Flux

# Zakładamy, że WriterVerifier.jl zawiera:
# - preprocess_image
# - create_model
# - binary_cross_entropy_loss
# - load_batch
# - create_pairs

push!(LOAD_PATH, "../src")
using WriterVerifier

@testset "Prosty test działania modelu" begin
    Random.seed!(123)

    # 1. Test przetwarzania obrazu
    @testset "preprocess_image" begin
        dummy_path = "nonexistent_file.png"
        img = preprocess_image(dummy_path; target_size=(32, 64))
        @test size(img) == (32, 64, 1)
        @test eltype(img) == Float32
    end

    # 2. Test konstrukcji modelu
    @testset "create_model" begin
        model = create_model((32, 64, 1))
        @test model isa SiameseNet
    end

    # 3. Test forward pass
    @testset "forward pass" begin
        model = create_model((32, 64, 1))
        x1 = rand(Float32, 32, 64, 1, 2)
        x2 = rand(Float32, 32, 64, 1, 2)
        output = model(x1, x2)
        @test size(output) == (2,)
        @test all(0 .<= output .<= 1)
    end

    # 4. Test straty
    @testset "binary_cross_entropy_loss" begin
        y_true = Float32[1.0, 0.0]
        y_pred = Float32[0.9, 0.1]
        loss = binary_cross_entropy_loss(y_pred, y_true)
        @test loss isa Float32
        @test loss > 0
    end
end
