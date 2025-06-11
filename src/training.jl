using Flux
using Flux: train!
using ProgressMeter
using Random
using Statistics
using JLD2



function train_model!(model, pairs, labels; 
                     epochs=10, 
                     batch_size=32,
                     learning_rate=1e-3,
                     train_split=0.8,
                     target_size=(64, 128),
                     loss_fn=binary_cross_entropy_loss,
                     device=cpu)
    """
    Trenuje model Siamese Network
    
    Args:
        model: SiameseNet model
        pairs: lista par obrazów
        labels: labels (1 = ten sam pisarz, 0 = różni pisarze)
        epochs: liczba epok
        batch_size: rozmiar batcha
        learning_rate: learning rate
        train_split: proporcja danych do treningu
        target_size: rozmiar obrazów (H, W)
        loss_fn: funkcja straty
        device: cpu lub gpu
    """

    n_samples = length(pairs)
    n_train = round(Int, n_samples * train_split)

    indices = shuffle(1:n_samples)
    train_indices = indices[1:n_train]
    val_indices = indices[n_train+1:end]

    model = model |> device

    optimizer = Adam(learning_rate)

    train_losses = Float32[]
    val_losses = Float32[]
    val_accuracies = Float32[]

    for epoch in 1:epochs

        model = train_mode!(model)
        train_loss = 0.0f0
        n_train_batches = 0

        train_batch_indices = shuffle(train_indices)

        @showprogress desc="Training" for i in 1:batch_size:length(train_batch_indices)
            batch_end = min(i + batch_size - 1, length(train_batch_indices))
            batch_idx = train_batch_indices[i:batch_end]

            x1, x2, y = load_batch(pairs, labels, batch_idx; target_size=target_size)
            x1, x2, y = x1 |> device, x2 |> device, y |> device

            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x1, x2)
                loss_fn(ŷ, y)
            end

            Flux.update!(optimizer, model, grads[1])

            train_loss += loss
            n_train_batches += 1
        end

        avg_train_loss = train_loss / n_train_batches
        push!(train_losses, avg_train_loss)

        model = eval_mode!(model)
        val_loss, val_acc = validate_model(model, pairs, labels, val_indices;
                                        batch_size=batch_size,
                                        target_size=target_size,
                                        loss_fn=loss_fn,
                                        device=device)
        push!(val_losses, val_loss)
        push!(val_accuracies, val_acc)


    end

    history = Dict(
        "train_losses" => train_losses,
        "val_losses" => val_losses,
        "val_accuracies" => val_accuracies
    )

    return model |> cpu, history
end


function validate_model(model, pairs, labels, val_indices;
                       batch_size=32, 
                       target_size=(64, 128),
                       loss_fn=binary_cross_entropy_loss,
                       device=cpu)
"""
Walidacja modelu
"""
    total_loss = 0.0f0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for i in 1:batch_size:length(val_indices)
        batch_end = min(i + batch_size - 1, length(val_indices))
        batch_idx = val_indices[i:batch_end]

        x1, x2, y = load_batch(pairs, labels, batch_idx; target_size=target_size)
        x1, x2, y = x1 |> device, x2 |> device, y |> device


        ŷ = model(x1, x2)

        loss = loss_fn(ŷ, y)
        total_loss += loss

        predictions = ŷ .> 0.5f0  # threshold dla binary classification
        correct = sum(predictions .== (y .> 0.5f0))
        total_correct += correct
        total_samples += length(y)
        n_batches += 1
    end

    avg_loss = total_loss / n_batches
    accuracy = total_correct / total_samples

    return avg_loss, accuracy
end

function train_mode!(model)
    Flux.trainmode!(model)
    return model
end

function eval_mode!(model)
    Flux.testmode!(model)
    return model
end

function save_model(model, filepath::String)
    @save filepath model
    println("Model saved to $filepath")
end

function load_model(filepath::String)
    @load filepath model
    println("Model loaded from $filepath")
    return model 
end

function plot_training_history(history)
    """
    Draws training plot
    """
    epochs = 1:length(history["train_losses"])

    p1 = plot(epochs, history["train_losses"], label="Train Loss", lw=2)
    plot!(p1, epochs, history["val_losses"], label="Val Loss", lw=2)
    xlabel!(p1, "Epoch")
    ylabel!(p1, "Loss")
    title!(p1, "Training and Validation Loss")

    p2 = plot(epochs, history["val_accuracies"] .* 100, label="Val Accuracy", lw=2, color=:green)
    xlabel!(p2, "Epoch")
    ylabel!(p2, "Accuracy (%)")
    title!(p2, "Validation Accuracy")

    return plot(p1, p2, layout=(2, 1), size=(800, 600))
end