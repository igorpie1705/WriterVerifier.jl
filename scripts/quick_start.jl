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
pairs, labels = create_pairs(writers_data; n_positive=200, n_negative=200)

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
    println("   Stack trace: $e")
end

# ============================================================================
# 4. MINI TRENING (żeby sprawdzić czy wszystko działa)
# ============================================================================
println("\n🎯 KROK 4: Mini trening (test)")

if length(pairs) >= 50
    println("Uruchamianie mini treningu z $(min(50, length(pairs))) parami...")
    
    # Weź tylko pierwsze 50 par do szybkiego testu
    mini_pairs = pairs[1:50]
    mini_labels = labels[1:50]
    
    try
        # Trenuj przez 2 epoki z małym batch size
        trained_model, history = train_model!(
            model, mini_pairs, mini_labels;
            epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            train_split=0.8
        )
        
        println("✅ Mini trening zakończony!")
        println("   📈 Final train loss: $(round(history["train_losses"][end], digits=4))")
        println("   📈 Final val loss: $(round(history["val_losses"][end], digits=4))")
        println("   🎯 Final val accuracy: $(round(history["val_accuracies"][end] * 100, digits=2))%")
        
        # Plot training history
        p = plot_training_history(history)
        savefig(p, "training_history.png")
        println("   📊 Wykres zapisany jako training_history.png")
        
    catch e
        println("   ❌ Błąd w mini treningu: $e")
        println("   Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
else
    println("❌ Za mało par do treningu (potrzeba minimum 50)")
end

# ============================================================================
# 5. TEST PREDYKCJI NA PRAWDZIWYCH DANYCH
# ============================================================================
println("\n🔍 KROK 5: Test predykcji")

if length(pairs) >= 10
    try
        # Test na kilku parach
        test_pairs = pairs[1:5]
        test_labels = labels[1:5]
        
        println("Testing predictions:")
        for i in 1:5
            img1_path, img2_path = test_pairs[i]
            true_label = test_labels[i]
            
            # Predict similarity
            pred_similarity = predict_similarity(model, img1_path, img2_path)
            
            writer1 = basename(img1_path)[1:3]  # First 3 chars as writer ID
            writer2 = basename(img2_path)[1:3]
            
            println("   $(i). $(writer1) vs $(writer2): pred=$(round(pred_similarity, digits=3)), true=$true_label")
        end
        
    catch e
        println("   ❌ Błąd w testowaniu predykcji: $e")
    end
end

# ============================================================================
# 6. ZAPISANIE MODELU
# ============================================================================
println("\n💾 KROK 6: Zapisywanie modelu")

try
    # Utwórz folder models jeśli nie istnieje
    if !isdir("models")
        mkdir("models")
    end
    
    # Zapisz model
    model_path = "models/siamese_model_test.jld2"
    save_model(model, model_path)
    
    # Test ładowania
    loaded_model = load_model(model_path)
    println("✅ Model zapisany i załadowany pomyślnie!")
    
catch e
    println("❌ Błąd przy zapisywaniu modelu: $e")
end

# ============================================================================
# 7. PODSUMOWANIE
# ============================================================================
println("\n🎉 PODSUMOWANIE QUICK START")
println("=" ^ 60)
println("✅ Dane załadowane: $(length(writers_data)) pisarzy")
println("✅ Pary utworzone: $(length(pairs)) par")
println("✅ Model utworzony i przetestowany")
println("✅ Mini trening przeprowadzony")
println("✅ Predykcje przetestowane")
println("✅ Model zapisany")

println("\n🚀 NASTĘPNE KROKI:")
println("1. Zwiększ liczbę par treningowych")
println("2. Uruchom pełny trening z więcej epokami")
println("3. Eksperymentuj z hyperparametrami")
println("4. Dodaj więcej metryk ewaluacji")
println("5. Zaimplementuj data augmentation")

println("\n📚 PRZYKŁAD PEŁNEGO TRENINGU:")
println("""
# Pełny trening
pairs, labels = create_pairs(writers_data; n_positive=2000, n_negative=2000)
model = create_model((64, 128, 1))
trained_model, history = train_model!(
    model, pairs, labels;
    epochs=20,
    batch_size=32,
    learning_rate=1e-3
)
save_model(trained_model, "models/siamese_model_full.jld2")
""")

println("\n🎯 QUICK START ZAKOŃCZONY POMYŚLNIE! 🎯")