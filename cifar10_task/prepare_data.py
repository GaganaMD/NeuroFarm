from resnet_model import resnet50
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms as T
import numpy as np


# Assume `model` is your pretrained ResNet-50 model
model = resnet50(pretrained=True)
# print(next(model.parameters()))
model.eval()


def extract_features(model, x):
    # Put the model in evaluation mode
    model.eval()

    # Forward pass through the network
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        representation = torch.flatten(x, 1)
        representation = model.fc(representation)

    return representation


def make_transform():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    # Step 1: Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize images as per ResNet-50 requirements
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return transform


def make_data():

    transform = make_transform()

    train_dataset = datasets.CIFAR10(
        root='data/', train=True, download=True, transform=transform)
    # Count occurrences of each class
    class_counts = {i: 0 for i in range(10)}  # CIFAR-10 has 10 classes
    subset_indices = []
    labels_vector = []

    # Iterate through the dataset and collect indices for 100 images per class
    for idx, (img, label) in enumerate(train_dataset):
        if class_counts[label] < 10:
            subset_indices.append(idx)
            labels_vector.append(label)  # Track the label of each image
            class_counts[label] += 1
        if all(count == 10 for count in class_counts.values()):
            break

    # Create a SubsetRandomSampler based on collected indices
    subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=subset_sampler)

    # Step 2: Embedding all images in the training set
    embeddings = []
    batch_size = 10  # Adjust batch size based on your system's memory

    # Ensure model is in evaluation mode

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = extract_features(model, inputs)
            embeddings.append(outputs)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} images")

    # Concatenate embeddings from all images into a single tensor
    embeddings_tensor = torch.cat(embeddings, dim=0)
    # Convert labels_vector to a numpy array for easy manipulation if needed
    labels_vector = np.array(labels_vector)

    # Concatenate embeddings from all batches into a single tensor
    embeddings_tensor = torch.cat(embeddings, dim=0)
    embeddings_tensor = embeddings_tensor.numpy()

    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(labels_vector.shape)

    # Save the embeddings and labels to disk
    np.save('logits/logits_cifar10.npy', embeddings_tensor)
    np.save('logits/labels_cifar10.npy', labels_vector)


if __name__ == "__main__":
    make_data()
