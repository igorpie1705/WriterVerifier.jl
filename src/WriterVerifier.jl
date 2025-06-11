module WriterVerifier

using Flux
using Images
using MLUtils
using FileIO
using Random
using StatsBase
using Plots
using ProgressMeter

# Eksport głównych funkcji
export load_iam_data, preprocess_image, create_pairs
export SiameseNet, create_model
export train_model!, evaluate_model
export contrastive_loss

# Include wszystkich plików
include("data_loader.jl") 
include("model.jl")
include("training.jl")

end # module
