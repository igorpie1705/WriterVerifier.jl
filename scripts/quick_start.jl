# ============================================================================
# QUICK START SCRIPT - Test całego pipeline'u
# ============================================================================
# Uruchom to po setup'ie projektu żeby przetestować wszystkie komponenty

using Pkg
Pkg.activate(".")  # Aktywuj projekt

# Załaduj nasz moduł
push!(LOAD_PATH, "./src")
using WriterVerifier

using Plots
using Random
Random.seed!(42)  # Dla reprodukowalności

println("🚀 QUICK START - HandwritingIdentification")
println("=" ^ 60)

# ============================================================================
# 1. ŁADOWANIE DANYCH
# ============================================================================
println("\n📁 KROK 1: Ładowanie danych IAM")

# ZMIEŃ TĘ ŚCIEŻKĘ NA SWOJĄ!
data_path = "data/raw/words"  # Tu powinny być Twoje obrazy IAM

if isdir(data_path)
    # Załaduj dane
    writers_data = load_iam_data(data_path; max_samples_per_writer=20)
    
    # Pokaż statystyki
    println("✅ Załadowano pisarzy: $(length(writers_data))")
    sample_writers = collect(keys(writers_data))[1:min(3, length(writers_data))]
    for writer in sample_writers
        println("   📝 $writer: $(length(writers_data[writer])) próbek")
    end
else
    println("❌ BŁĄD: Nie znaleziono folderu $data_path")
    println("   Skopiuj dataset IAM Words do tego foldera i uruchom ponownie")
    exit(1)
end

# ============================================================================
# 2. PREPROCESSING I PARY
# ============================================================================
println("\n🔄 KROK 2: Tworzenie par treningowych")

# Utwórz pary do treningu
pairs, labels = create_pairs(writers_data; n_positive=100, n_negative=100)

println("✅ Utworzono $(length(pairs)) par")
println("   Pozytywne pary: $(sum(labels))")
println("   Negatywne pary: $(length(labels) - sum(labels))")

# Test preprocessing na pierwszej parze
println("\n🖼️  Test preprocessing obrazów...")
img1_path, img2_path = pairs[1]
println("   Przetwarzanie: $(basename(img1_path))")

try
    img1 = preprocess_image(img1_path; target_size=(64, 128))
    img2 = preprocess_image(img2_path; target_size=(64, 128))
    
    println("   ✅ Rozmiar obrazu 1: $(size(img1))")
    println("   ✅ Rozmiar obrazu 2: $(size(img2))")
    println("   ✅ Typ danych: $(eltype(img1))")
    println("   ✅ Zakres wartości: [$(minimum(img1)), $(maximum(img1))]")
catch e
    println("   ❌ Błąd preprocessing: $e")
end

# ============================================================================
# 3. MODEL
# ============================================================================
println("\n🏗️  KROK 3: Tworzenie modelu")

# Utwórz model
model = create_model((64, 128, 1))
println("✅ Model utworzony")

# Pokaż architekturę
model_summary(model, (64, 128, 1, 4))

# Test forward pass
println("\n🧪 Test forward pass...")
try
    # Utwórz przykładowe dane
    x1 = randn(Float32, 64, 128, 1, 2)
    x2 = randn(Float32, 64, 128, 1, 2) 
    
    # Forward pass
    output = model(x1, x2)
    println("   ✅ Output shape: $(size(output))")
    println("   ✅ Output values: $output")
    println("   ✅ Output range: [$(minimum(output)), $(maximum(output))]")
    
    # Test loss function
    y_true = Float32[1, 0]  # przykładowe labels
    loss = binary_cross_entropy_loss(output, y_true)
    println("   ✅ Loss value: $loss")
    
catch e
    println("   ❌ Błąd w forward pass: $e")
end

# ============================================================================
# 4. MINI TRENING (żeby sprawdzić czy wszystko działa)
# ============================================================================
println("\n🎯 KROK 4: Mini trening (test)")

if length(pairs) >= 50
    println("Uruchamianie mini treningu z $(min(50, length(pairs))) parami...")
    
    # Weź tylko pierwsze
end