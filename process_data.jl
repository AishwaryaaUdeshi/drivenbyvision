using Flux
using Images
using ImageMagick

# Function to load the images from the folder
function load_images(folder_path::AbstractString)
    images = []
    for file in readdir(folder_path)
        # Check if the file is a PNG image
        if splitext(file)[2] == ".png"
            img_path = joinpath(folder_path, file)
            img = load(img_path)
            push!(images, img)
        end
    end
    return images
end

# Function to load the labels from the folder
function load_labels(folder_path::AbstractString)
    labels = []
    for file in readdir(folder_path)
        # Check if the file is a PNG image
        if splitext(file)[2] == ".png"
            label_path = joinpath(folder_path, file)
            label = load(label_path)
            push!(labels, label)
        end
    end
    return labels
end

# Load the images and labels for training set
train_image_folder = "train_set/clips/0313-1"
train_label_folder = "train_set/seg_label/0313-1"

train_images = load_images(train_image_folder)
train_labels = load_labels(train_label_folder)

# Load the images for the test set
test_image_folder = "test_set/clips/0530"
test_images = load_images(test_image_folder)

# Verify the number of loaded images and labels
println("Number of training images: ", length(train_images))
println("Number of training labels: ", length(train_labels))
println("Number of test images: ", length(test_images))