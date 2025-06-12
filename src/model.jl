using Flux
using Functors
struct SiameseNetwork
    feature_net
    similarity_net
end

Functors.@functor SiameseNetwork

"""
Simple CNN for extracting image features
"""
function create_feature_net()
    return Chain(
        Conv((5, 5), 1 => 32, relu, pad=2),
        MaxPool((2, 2)),
        
        Conv((3, 3), 32 => 64, relu, pad=1),
        MaxPool((2, 2)),
        
        Conv((3, 3), 64 => 128, relu, pad=1),
        MaxPool((2, 2)),
        
        Flux.flatten,
        
        Dense(128 * 8 * 16, 256, relu),
        Dropout(0.5),
        Dense(256, 128)
    )
end

"""
Simple network for calculating similarity from feature difference
"""
function create_similarity_net()
    return Chain(
        Dense(128, 64, relu),
        Dropout(0.3),
        Dense(64, 32, relu),
        Dense(32, 1, sigmoid)  # Output 0-1: 0=different, 1=same
    )
end

"""
Creates complete Siamese model
"""
function create_model()
    feature_net = create_feature_net()
    similarity_net = create_similarity_net()
    
    return SiameseNetwork(feature_net, similarity_net)
end

"""
Model forward pass:
1. Extract features from both images
2. Calculate absolute difference
3. Pass through similarity network
"""
function (model::SiameseNetwork)(x1, x2)
    features1 = model.feature_net(x1)
    features2 = model.feature_net(x2)
    
    difference = abs.(features1 .- features2)
    
    similarity = model.similarity_net(difference)
    
    return dropdims(similarity, dims=1)
end

"""
Simple loss function - binary cross entropy
"""
function loss_function(y_pred, y_true)
    epsilon = 1f-7
    y_pred = clamp.(y_pred, epsilon, 1.0f0 - epsilon)
    
    return -mean(y_true .* log.(y_pred) .+ (1.0f0 .- y_true) .* log.(1.0f0 .- y_pred))
end