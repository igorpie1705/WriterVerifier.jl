using Images, FileIO, Random

"""
Loads images from folder
"""
function load_images(folder_path; max_per_writer=20)
    writers = Dict{String, Vector{String}}()
    
    if !isdir(folder_path)
        println("Folder $folder_path doesn't exist!")
        return writers
    end
    
    # Find all images
    for (root, dirs, files) in walkdir(folder_path)
        for file in files
            if any(endswith(lowercase(file), ext) for ext in [".png", ".jpg", ".jpeg"])
                full_path = joinpath(root, file)

                parts = split(file, "-")
                if length(parts) >= 2
                    writer = parts[1]

                    if !haskey(writers, writer)
                        writers[writer] = String[]
                    end

                    if length(writers[writer]) < max_per_writer
                        push!(writers[writer], full_path)
                    end
                end
            end
        end
    end
    
    # Remove writers with less than 2 images
    writers = filter(p -> length(p.second) >= 2, writers)
    
    println("Loaded $(length(writers)) writers")
    for (writer, images) in writers
        println("$writer: $(length(images)) images")
    end
    
    return writers
end

"""
Processes single image for network input
"""
function process_image(path; size=(64, 64))
    try
        img = load(path)
        img = Gray.(img)
        img = imresize(img, size)
        
        # Convert to Float32 and normalize
        img = Float32.(img)
        img = 2.0f0 * img .- 1.0f0  # Normalize to [-1, 1]
        
        # Add channel dimension
        return reshape(img, size..., 1)
        
    catch e
        @warn "Error processing $path: $e"
        return zeros(Float32, size..., 1)
    end
end

"""
Creates image pairs for training
positive = same writer pairs (label = 1)
negative = different writer pairs (label = 0)
"""
function create_pairs(writers; positive=100, negative=100)
    pairs = []
    labels = []
    
    writer_list = collect(keys(writers))
    
    # Positive pairs (same writer)
    println("Creating positive pairs...")
    for _ in 1:positive
        writer = rand(writer_list)
        if length(writers[writer]) >= 2
            img1, img2 = rand(writers[writer], 2)
            if img1 != img2
                push!(pairs, (img1, img2))
                push!(labels, 1)
            end
        end
    end
    
    # Negative pairs (different writers)
    println("Creating negative pairs...")
    for _ in 1:negative
        if length(writer_list) >= 2
            writer1, writer2 = rand(writer_list, 2)
            if writer1 != writer2
                img1 = rand(writers[writer1])
                img2 = rand(writers[writer2])
                push!(pairs, (img1, img2))
                push!(labels, 0)
            end
        end
    end
    
    println("Created $(length(pairs)) pairs")
    println("Positive: $(sum(labels))")
    println("Negative: $(length(labels) - sum(labels))")
    
    return pairs, labels
end

"""
Loads a batch of data
"""
function load_batch(pairs, labels, indices; image_size=(64, 64))
    batch_size = length(indices)
    
    x1 = zeros(Float32, image_size..., 1, batch_size)
    x2 = zeros(Float32, image_size..., 1, batch_size)
    y = zeros(Float32, batch_size)
    
    for (i, idx) in enumerate(indices)
        img1_path, img2_path = pairs[idx]
        
        x1[:, :, :, i] = process_image(img1_path; size=image_size)
        x2[:, :, :, i] = process_image(img2_path; size=image_size)
        y[i] = Float32(labels[idx])
    end
    
    return x1, x2, y
end