from torchvision import datasets, transforms

# Load MNIST with no augmentation
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Your indices
poison_images_test = [330, 568, 3934, 12336, 30560]
poison_images = [30696, 33105, 33615, 33907, 36848, 40713, 41706]

# Check test dataset labels
print("Test set labels:")
for idx in poison_images_test:
    print(f"Index {idx}: Label {train_dataset[idx][1]}")

# Check train dataset labels
print("\nTrain set labels:")
for idx in poison_images:
    print(f"Index {idx}: Label {train_dataset[idx][1]}")
