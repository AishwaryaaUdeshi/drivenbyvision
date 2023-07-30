using Flux
using Images  # For loading and resizing images
using Statistics
using Random
import Pkg
Pkg.add("BSON")
using BSON

# Load the model from the .bson file
model_path = "model.bson"
model = BSON.load(model_path)[:model]

using Flux
using Images
using ImageMagick
using Statistics
using Random

# Define the model architecture in Julia
model = Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), _=>16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(prod(12, 12, 16), 120, relu),  # Modify the input size to match the shape after MaxPool layers
    Dense(120, 84, relu),
    ConvTranspose((3, 3), 84 => 1, relu),  # Final ConvTranspose layer with 1 output channel
)

# Load the model weights from a BSON file
model_path = "model.bson"
model = BSON.load(model_path)[:model]

# Class to average lanes with
struct Lanes
    recent_fit::Vector{Array{Float32, 2}}
    avg_fit::Array{Float32, 2}
end

function Lanes()
    Lanes([], zeros(Float32, 80, 160))
end

# Resize the image and predict the lane to be drawn from the model in G color
function road_lines_image(image_path::String, lanes::Lanes)
    # Load and resize the image
    img = load(image_path)
    actual_image = imresize(img, (720, 1280, 3))

    # Get image ready for feeding into the model
    small_img = Flux.Data.DataLoader(img, batchsize=1) |> first

    # Make prediction with the neural network (un-normalize value by multiplying by 255)
    prediction = model(small_img)[1] * 255

    # Add lane prediction to the list for averaging
    push!(lanes.recent_fit, prediction)

    # Only using last five for average
    if length(lanes.recent_fit) > 5
        popfirst!(lanes.recent_fit)
    end

    # Calculate average detection
    lanes.avg_fit = mean(hcat(lanes.recent_fit...), dims=2)

    # Generate fake R & B color dimensions, stack with G
    blanks = zeros(Float32, 80, 160)
    lane_drawn = cat(blanks, lanes.avg_fit, blanks, dims=3)

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (720, 1280, 3))

    # Merge the lane drawing onto the original image
    result = clamp.(lane_image .+ actual_image, 0, 1)

    return result
end

# Create a lanes object
lanes = Lanes()

# Loop through the test images and predict the lane detector
for path in glob("test_images/*.jpg")
    res_img = road_lines_image(path, lanes)
    out_path = "test_predict/$(basename(path))"
    save(out_path, res_img)
end
