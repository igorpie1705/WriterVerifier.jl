using Flux
using Random
using Statistics
using JLD2


"""
Model training function
"""
function train_model!(model, pairs, labels; 
                      epochs=5, 
                      batch_size=16,
                      learning_rate=0.001)
    println("Starting training...")
    
    n_total = length(pairs)
    n_train = round(Int, n_total * 0.8)
    
    indices = shuffle(1:n_total)
    train_idx = indices[1:n_train]
    val_idx = indices[n_train+1:end]
    
    println("Training data: $n_train")
    println("Validation data: $(length(val_idx))")
    
    optim = Adam(learning_rate)
    opt_state = Flux.setup(optim, model)
    
    train_history = Float32[]
    val_history = Float32[]
    acc_history = Float32[]
    
    for epoch in 1:epochs
        println("\nEpoch $epoch/$epochs")
        
        Flux.trainmode!(model)
        train_losses = Float32[]
        
        shuffled_train = shuffle(train_idx)
        
        for i in 1:batch_size:length(shuffled_train)
            end_idx = min(i + batch_size - 1, length(shuffled_train))
            batch_idx = shuffled_train[i:end_idx]
            
            x1, x2, y = load_batch(pairs, labels, batch_idx)
            
            loss, grads = Flux.withgradient(model) do m
                y_pred = m(x1, x2)
                loss_function(y_pred, y)
            end
            
            Flux.update!(opt_state, model, grads[1])
            
            push!(train_losses, loss)
        end
        
        Flux.testmode!(model)
        val_loss, val_acc = evaluate_model(model, pairs, labels, val_idx)
        
        push!(train_history, mean(train_losses))
        push!(val_history, val_loss)
        push!(acc_history, val_acc)
        
        println("Train loss: $(round(mean(train_losses), digits=4))")
        println("Val loss: $(round(val_loss, digits=4))")
        println("Val accuracy: $(round(val_acc * 100, digits=1))%")
    end
    
    println("\nTraining completed!")
    
    history = Dict(
        "train_loss" => train_history,
        "val_loss" => val_history,
        "val_acc" => acc_history
    )
    
    return model, history
end

"""
Evaluates model on validation data
"""
function evaluate_model(model, pairs, labels, val_idx; batch_size=32)
    losses = Float32[]
    correct = 0
    total = 0
    
    for i in 1:batch_size:length(val_idx)
        end_idx = min(i + batch_size - 1, length(val_idx))
        batch_idx = val_idx[i:end_idx]
        
        x1, x2, y = load_batch(pairs, labels, batch_idx)
        
        y_pred = model(x1, x2)
        
        loss = loss_function(y_pred, y)
        push!(losses, loss)
        
        predictions = y_pred .> 0.5f0
        truth = y .> 0.5f0
        correct += sum(predictions .== truth)
        total += length(y)
    end
    
    return mean(losses), correct / total
end

"""
Tests similarity between two images
"""
function test_similarity(model, path1, path2)
    try
        img1 = process_image(path1)
        img2 = process_image(path2)
        
        x1 = reshape(img1, size(img1)..., 1)
        x2 = reshape(img2, size(img2)..., 1)
        
        Flux.testmode!(model)
        similarity = model(x1, x2)[1]
        
        return Float32(similarity)
    catch e
        @warn "Error in testing: $e"
        return 0.0f0
    end
end

"""
Saves model to file
"""
function save_model(model, path)
    @save path model
    println("Model saved: $path")
end

"""
Loads model from file
"""
function load_model(path)
    @load path model
    println("Model loaded: $path")
    return model
end