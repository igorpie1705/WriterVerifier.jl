using Flux
using Functors: @functor


struct SiameseNet
    backbone::Chain
    distance_head::Chain
end

@functor SiameseNet

function create_cnn_backbone(input_channels=1)

    return Chain(
        Conv((5, 5), input_channels => 32, relu, pad=2),
        BatchNorm(32),
        MaxPool((2, 2)),

        Conv((3, 3), 32 => 64, relu, pad=1),
        BatchNorm(64),
        MaxPool((2, 2)),

        Conv((3, 3), 64 => 128, relu, pad=1),
        BatchNorm(128),
        MaxPool((2, 2)),

        Conv((3, 3), 128 => 256, relu, pad=1),
        BatchNorm(256),
        AdaptiveMeanPool((4, 4)),

        Flux.flatten,
        Dense(256 * 4 * 4, 512, relu),
        Dropout(0.5),
        Dense(512, 256, relu),
        Dropout(0.3),
        Dense(256, 128)
    )
end

function create_distance_head()
    """
    Tworzy kompletny model Siamese Network
    """
    return Chain(
        Dense(1, 64, relu),
        Dropout(0.2),
        Dense(64, 32, relu),
        Dense(32, 1, sigmoid)
    )
end


function create_model(input_size=(64, 128, 1))
    backbone = create_cnn_backbone(input_size[3])
    distance_head = create_distance_head() 

    return SiameseNet(backbone, distance_head)
end


function (m::SiameseNet)(x1, x2)

    features1 = m.backbone(x1)
    features2 = m.backbone(x2)

    l1_distance = sum(abs.(features1 .- features2), dims=1)

    similarity = m.distance_head(l1_distance)

    return dropdims(similarity, dims=1)

end

function contrastive_loss(ŷ, y; margin=1.0f0)

    distances = 1.0f0 .- ŷ

    positive_loss = y .* (distances .^ 2)
    negative_loss = (1.0f0 .- y) .* max.(0.0f0, margin .- distances) .^ 2

    return mean(0.5f0 * (positive_loss .+ negative_loss))
end

function binary_cross_entropy_loss(ŷ, y)
    ϵ = 1f-7
    ŷ_clipped = clamp.(ŷ, ϵ, 1.0f0 - ϵ)
    return -mean(y .* log.(ŷ_clipped) .+ (1.0f0 .- y) .* log.(1.0f0 .- ŷ_clipped))
end

function model_summary(model::SiameseNet, input_size=(64, 128, 1, 4))

    x1 = randn(Float32, input_size...)
    x2 = randn(Float32, input_size...)

    features1 = model.backbone(x1)

    output = model(x1, x2)

    total_params = sum(length, Flux.params(model))


    return nothing

end