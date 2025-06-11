module WriterVerifier

using Flux
using Images
using MLUtils
using FileIO
using Random
using StatsBase
using Plots
using ProgressMeter
using Statistics
using JLD2

# Eksport głównych funkcji
export load_iam_data, preprocess_image, create_pairs, load_batch
export SiameseNet, create_model, predict_similarity
export train_model!, evaluate_model, validate_model
export contrastive_loss, binary_cross_entropy_loss
export model_summary
export save_model, load_model, plot_training_history
export train_mode!, eval_mode!

# Include wszystkich plików
include("data_loader.jl") 
include("model.jl")
include("training.jl")

end # module