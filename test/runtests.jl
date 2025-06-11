using Test
using WriterVerifier  # Assuming the module is in your current namespace
using Images

# Create a temporary directory for testing
const TEST_DIR = mktempdir()
const IMG_SIZE = (64, 64)

function setup_test_images()
    # Create 2 test writers with 3 sample images each
    for writer in ["writer1", "writer2"]
        mkdir(joinpath(TEST_DIR, writer))
        for i in 1:3
            # Create simple gradient images
            img = rand(Gray{Float32}, IMG_SIZE)
            save(joinpath(TEST_DIR, "$(writer)_$i.png"), img)
        end
    end
end

@testset "WriterVerifier Tests" begin
    # Setup test environment
    @testset "Setup" begin
        setup_test_images()
        @test isdir(TEST_DIR)
        @test length(readdir(TEST_DIR)) == 2
    end

    @testset "Data Loading" begin
        writers = load_images(TEST_DIR)
        @test length(writers) == 2
        @test all(x -> length(x) >= 2, values(writers))
        
        img = process_image(first(values(writers))[1])
        @test size(img) == (IMG_SIZE..., 1)
        @test eltype(img) == Float32
        
        pairs, labels = create_pairs(writers; positive=5, negative=5)
        @test length(pairs) == length(labels)
        @test sum(labels) == 5  # 5 positive pairs
        
        x1, x2, y = load_batch(pairs, labels, 1:2)
        @test size(x1) == (IMG_SIZE..., 1, 2)
    end

    @testset "Model" begin
        model = create_model()
        @test model isa SimpleSiamese
        
        # Test forward pass
        x = rand(Float32, IMG_SIZE..., 1, 2)
        output = model(x, x)
        @test size(output) == (2,)
        @test all(0 .<= output .<= 1)
        
        # Test loss function
        loss = loss_function([0.6f0, 0.4f0], [1f0, 0f0])
        @test loss > 0
        
        # Test model evaluation
        test_result = test_model(model)
        @test test_result
    end

    @testset "Training" begin
        writers = load_images(TEST_DIR)
        pairs, labels = create_pairs(writers; positive=5, negative=5)
        model = create_model()
        
        # Smoke test training
        trained_model, history = train_model!(
            model, pairs, labels;
            epochs=1, batch_size=2
        )
        @test trained_model isa SimpleSiamese
        @test haskey(history, "train_loss")
        
        # Test similarity function
        test_img1 = first(values(writers))[1]
        test_img2 = first(values(writers))[2]
        similarity = test_similarity(model, test_img1, test_img2)
        @test 0 <= similarity <= 1
        
        # Test model saving/loading
        test_model_path = joinpath(TEST_DIR, "test_model.jld2")
        save_model(model, test_model_path)
        @test isfile(test_model_path)
        
        loaded_model = load_model(test_model_path)
        @test loaded_model isa SimpleSiamese
    end

    # Cleanup
    rm(TEST_DIR; recursive=true)
end

println("\n✅ All tests passed!")