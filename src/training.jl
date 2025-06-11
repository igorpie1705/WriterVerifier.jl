using Flux
using Random
using Statistics
using JLD2


function train_model!(model, pairs, labels; 
                      epochs=5, 
                      batch_size=16,
                      learning_rate=0.001)
    """
    Simple training function - without unnecessary complications
    """
    println("Starting training...")
    
    # Split data into training and validation
    n_total = length(pairs)
    n_train = round(Int, n_total * 0.8)
    
    indices = shuffle(1:n_total)
    train_idx = indices[1:n_train]
    val_idx = indices[n_train+1:end]
    
    println("Training data: $n_train")
    println("Validation data: $(length(val_idx))")
    
    # Prepare optimizer
    optim = Adam(learning_rate)
    opt_state = Flux.setup(optim, model)
    
    # Training history
    train_history = Float32[]
    val_history = Float32[]
    acc_history = Float32[]
    
    # Main training loop
    for epoch in 1:epochs
        println("\nEpoch $epoch/$epochs")
        
        # === TRAINING ===
        Flux.trainmode!(model)
        train_losses = Float32[]
        
        # Shuffle training data
        shuffled_train = shuffle(train_idx)
        
        # Train in batches
        for i in 1:batch_size:length(shuffled_train)
            # Determine batch range
            end_idx = min(i + batch_size - 1, length(shuffled_train))
            batch_idx = shuffled_train[i:end_idx]
            
            # Load batch
            x1, x2, y = load_batch(pairs, labels, batch_idx)
            
            # Compute gradients and loss
            loss, grads = Flux.withgradient(model) do m
                y_pred = m(x1, x2)
                loss_function(y_pred, y)
            end
            
            # Update weights
            Flux.update!(opt_state, model, grads[1])
            
            push!(train_losses, loss)
        end
        
        # === VALIDATION ===
        Flux.testmode!(model)
        val_loss, val_acc = evaluate_model(model, pairs, labels, val_idx)
        
        # Save history
        push!(train_history, mean(train_losses))
        push!(val_history, val_loss)
        push!(acc_history, val_acc)
        
        # Display results
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

function evaluate_model(model, pairs, labels, val_idx; batch_size=32)
    """
    Evaluates model on validation data
    """
    losses = Float32[]
    correct = 0
    total = 0
    
    for i in 1:batch_size:length(val_idx)
        end_idx = min(i + batch_size - 1, length(val_idx))
        batch_idx = val_idx[i:end_idx]
        
        # Load batch
        x1, x2, y = load_batch(pairs, labels, batch_idx)
        
        # Prediction
        y_pred = model(x1, x2)
        
        # Loss
        loss = loss_function(y_pred, y)
        push!(losses, loss)
        
        # Accuracy
        predictions = y_pred .> 0.5f0
        truth = y .> 0.5f0
        correct += sum(predictions .== truth)
        total += length(y)
    end
    
    return mean(losses), correct / total
end

function test_similarity(model, path1, path2)
    """
    Tests similarity between two images
    """
    try
        # Process images
        img1 = process_image(path1)
        img2 = process_image(path2)
        
        # Add batch dimension
        x1 = reshape(img1, size(img1)..., 1)
        x2 = reshape(img2, size(img2)..., 1)
        
        # Prediction
        Flux.testmode!(model)
        similarity = model(x1, x2)[1]
        
        return Float32(similarity)
    catch e
        @warn "Error in testing: $e"
        return 0.0f0
    end
end

function save_model(model, path)
    """
    Saves model to file
    """
    @save path model
    println("Model saved: $path")
end

function load_model(path)
    """
    Loads model from file
    """
    @load path model
    println("Model loaded: $path")
    return model
end