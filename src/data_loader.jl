using Images, FileIO, MLUtils, Random


"""
Ładuje i przetwarza dane z IAM Words Dataset
"""

function load_iam_data(data_path::String; max_samples_per_writer=50)
    
    image_files = []
    for (root, dirs, files) in walkdir(data_path)
        for file in files
            if any(s -> endswith(lowercase(file), s), [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
                push!(image_files, joinpath(root, file))
            end
        end
    end

    println("Znaleziono $(length(image_files)) obrazów")

    writers_data = Dict{String, Vector{String}}()

    for img_path in image_files
        filename = basename(img_path)

        writer_match = match(r"^([a-zA-Z0-9]+)", filename)

        if writer_match !== nothing
            writer_id = writer_match.captures[1]

            if !haskey(writers_data, writer_id)
                writers_data[writer_id] = String[]
            end

            if length(writers_data[writer_id]) < max_samples_per_writer
                push!(writers_data[writer_id], img_path)
            end
        end
    end

    min_samples = 5
    writers_data = filter(p -> length(p.second) >= min_samples, writers_data)

    total_samples = sum(length(samples) for samples in values(writers_data))
    println("Załadowano $(length(writers_data)) pisarzy z $(total_samples) próbkami)")
    println("Średnio $(round(total_samples/length(writers_data), digits=1))) próbek na pisarza")

    return writers_data
end


function preprocess_image(img_path::String; target_size=(64, 128))

    try

        img = load(img_path)

        # Konwersja do skali szarości
        if ndims(img) == 3 || eltype(img) <: RGB
            img = Gray.(img)
        end

        img = Float32.(img)

        img_resized = imresize(img, target_size)

        img_normalized = 2.0f0 * img_resized .- 1.0f0

        return reshape(img_normalized, size(img_normalized)..., 1)

    catch e
        @warn "Błąd przy przetwarzaniu $img_path: $e"

        return zeros(Float32, target_size..., 1)
    end
end

function create_pairs(writers_data::Dict{String, Vector{String}};
                      n_positive=1000, n_negative=1000)
        
    pairs = Tuple{String, String}[]
    labels = Int[]

    writers = collect(keys(writers_data))

    positive_count = 0
    while positive_count < n_positive

        writer = rand(writers)
        writer_images = writers_data[writer]

        if length(writer_images) >= 2
            img1, img2 = rand(writer_images, 2)
            push!(pairs, (img1, img2))
            push!(labels, 1)
            positive_count += 1
        end
    end

    negative_count = 0
    while negative_count < n_negative

        writer1, writer2 = rand(writers, 2)
        if writer1 != writer2
            img1 = rand(writers_data[writer1])
            img2 = rand(writers_data[writer2])
            push!(pairs, (img1, img2))
            push!(labels, 0)
            negative_count += 1
        end
    end

    println("Utworzono $(length(pairs)) par ($(n_positive) pozytywnych, $(n_negative) negatywnych)")

    return pairs, labels
end

function load_batch(pairs, labels, batch_indices; target_size=(64, 128))

    batch_x1 = zeros(Float32, target_size..., 1, batch_size)
    batch_x2 = zeros(Float32, target_size..., 1, batch_size)
    batch_y = zeros(Float32, batch_size)

    for (i, idx) in enumerate(batch_indices)
        img1_path, img2_path = pairs[idx]

        batch_x1[:, :, :, i] = preprocess_image(img1_path; target_size=target_size)
        batch_x2[:, :, :, i] = preprocess_image(img2_path; target_size=target_size)
        batch_y[i] = Float32(labels[idx])
    end

    return batch_x1, batch_x2, batch_y
end