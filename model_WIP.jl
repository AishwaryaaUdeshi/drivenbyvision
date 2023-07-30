using Flux
using Statistics
using Random

# Load training images
train_images = load("train_images.p")  

# Load image labels
labels = load("train_labels.p")  

# Make into arrays as the neural network wants these
train_images = permutedims(Array(train_images), (4, 3, 2, 1))
labels = permutedims(Array(labels), (4, 3, 2, 1))

# Normalize labels - training images get normalized to start in the network
labels = labels ./ 255

# Shuffle images along with their labels
train_images, labels = shuffle(train_images, labels, dims=4)

# Split into training/validation sets
X_train, X_val, y_train, y_val = Flux.Data.train_test_split(train_images, labels, test_size=0.1)

# Batch size, epochs, and pool size below are all parameters to fiddle with for optimization
batch_size = 150
epochs = 20
pool_size = (2, 2)
input_shape = size(X_train)[2:end]

# Define the model (Modified lenet architecture)
model = Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool(pool_size),
    Conv((5, 5), _=>16, relu),
    MaxPool(pool_size),
    Flux.flatten,
    Dense(prod(size(X_train)[1:end-2])*16, 120, relu),  # Flatten before Dense layer
    Dense(120, 84, relu),
    ConvTranspose((3, 3), 84 => 1, relu),  # Final ConvTranspose layer with 1 output channel
)

# Summary of the model
println("Model Summary:")
summary(model)

# Data augmentation - If needed, you can use ImageTransformations.jl for image augmentation

# Compiling and training the model
opt = Flux.ADAM()
loss(x, y) = Flux.mse(model(x), y)
data = Flux.Data.DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)

println("Training...")

for epoch in 1:epochs
    for (x, y) in data
        Flux.train!(loss, Flux.params(model), [(x, y)], opt)
    end
    println("Epoch $epoch completed. Loss: $(loss(X_train, y_train))")
end

# Save model architecture and weights
using BSON
model_path = "model.bson"
BSON.@save model_path model

println("Model saved as $model_path")
