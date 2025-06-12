module WriterVerifier

export load_images, process_image, create_pairs, load_batch, SiameseNetwork
export create_feature_net, create_similarity_net, create_model, loss_function
export train_model!, evaluate_model, test_similarity, save_model, load_model

include("data_loader.jl")
include("model.jl") 
include("training.jl")

end # module