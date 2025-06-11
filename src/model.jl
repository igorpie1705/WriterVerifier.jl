using Flux
using Functors
# Siamese model structure
struct SiameseNetwork
    feature_net      # CNN for feature extraction
    similarity_net   # Network for calculating similarity
end

# Tell Flux to treat our struct as a model
Functors.@functor SiameseNetwork

function create_feature_net()
    """
    Simple CNN for extracting image features
    """
    return Chain(
        # First layer - basic features
        Conv((5, 5), 1 => 32, relu, pad=2),
        MaxPool((2, 2)),
        
        # Second layer - more complex features  
        Conv((3, 3), 32 => 64, relu, pad=1),
        MaxPool((2, 2)),
        
        # Third layer - high-level features
        Conv((3, 3), 64 => 128, relu, pad=1),
        MaxPool((2, 2)),
        
        # Flatten to vector
        Flux.flatten,
        
        # Dense layers - final representation
        Dense(128 * 8 * 8, 256, relu),
        Dropout(0.5),
        Dense(256, 128)  # Final feature vector
    )
end

function create_similarity_net()
    """
    Simple network for calculating similarity from feature difference
    """
    return Chain(
        Dense(128, 64, relu),
        Dropout(0.3),
        Dense(64, 32, relu),
        Dense(32, 1, sigmoid)  # Output 0-1: 0=different, 1=same
    )
end

function create_model()
    """
    Creates complete Siamese model
    """
    feature_net = create_feature_net()
    similarity_net = create_similarity_net()
    
    return SiameseNetwork(feature_net, similarity_net)
end

# How the model should work
function (model::SiameseNetwork)(x1, x2)
    """
    Model forward pass:
    1. Extract features from both images
    2. Calculate absolute difference
    3. Pass through similarity network
    """
    # Extract features from both images (using same network!)
    features1 = model.feature_net(x1)
    features2 = model.feature_net(x2)
    
    # Calculate absolute difference
    difference = abs.(features1 .- features2)
    
    # Calculate similarity
    similarity = model.similarity_net(difference)
    
    return dropdims(similarity, dims=1)  # Remove unnecessary dimension
end

function loss_function(y_pred, y_true)
    """
    Simple loss function - binary cross entropy
    """
    epsilon = 1f-7  # To avoid log(0)
    y_pred = clamp.(y_pred, epsilon, 1.0f0 - epsilon)
    
    return -mean(y_true .* log.(y_pred) .+ (1.0f0 .- y_true) .* log.(1.0f0 .- y_pred))
end