using Test
using WriterVerifier
using Images
using FileIO

# Tests for data_loader.jl
@testset "All tests" begin
    @testset "Data loading tests" begin

        test_dir = "data/test_data"
        writers = load_images(test_dir);
        println(writers)
        @test length(writers) == 2 # Should find two writers
        
        # Testing image processing
        if length(writers) > 0
            first_writer = first(keys(writers))
            img_path = writers[first_writer][1]
            processed = process_image(img_path);
            @test size(processed) == (64, 128, 1)
            
            # Testing pair creation
            pairs, labels = create_pairs(writers, positive=5, negative=5)
            @test length(pairs) == length(labels)
            @test sum(labels) > 0
        end

    end

    # Tests for model.jl
    @testset "Model tests" begin
        model = create_model()
        
        x1 = rand(Float32, 64, 128, 1, 1)
        x2 = rand(Float32, 64, 128, 1, 1)
        output = model(x1, x2)
        @test size(output) == (1,)
        @test 0 <= output[1] <= 1
        
        y_pred = [0.2f0, 0.8f0]
        y_true = [0.0f0, 1.0f0]
        loss = loss_function(y_pred, y_true);
        @test loss > 0
    end

    # Tests for training.jl
    @testset "Training tests" begin
        test_dir = "data/test_data"

        writers = load_images(test_dir)
        pairs, labels = create_pairs(writers, positive=5, negative=5);
        
        model = create_model()
        
        val_loss, val_acc = evaluate_model(model, pairs, labels, 1:length(pairs), batch_size=2);
        @test val_loss > 0
        @test 0 <= val_acc <= 1
        
        model_path = joinpath(test_dir, "test_model.jld2")
        save_model(model, model_path)
        @test isfile(model_path)
        
        loaded_model = load_model(model_path)
        @test loaded_model isa SiameseNetwork    
    end
end
