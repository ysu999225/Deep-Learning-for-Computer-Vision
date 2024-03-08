#for this question, we only consider the ResNet-18 Model
from torchvision.models import resnet18, ResNet18_Weights

# Initialize the pretrained ResNet-18 model
resnet18 =  torchvision.models.resnet18(pretrained=True)
resnet18



# Unfreeze the entire model
#for param in resnet18.parameters():
    #param.requires_grad = True
#"freeze" the weights of a model,
for param in resnet18.parameters():
    param.requires_grad = False

#use the nn.Sequential to define the model as above
class Resnet_model(nn.Module):
    def __init__(self):
        super(Resnet_model, self).__init__()
        
        
        self.model = nn.Sequential(
            nn.Linear(512, 120),  
            nn.ReLU(),           
            nn.Linear(120, 10)   
        )

    def forward(self, x):
      x = x.reshape(x.size(0),-1)
      x = self.model(x)
      return x





#check the features and bias are True
model_ = Resnet_model().to(device)
model_

model.fc = Resnet_model().to(device)
# Print the modified resnet18 model
print(model)


#random cropping, random flipping, and random color jittering to the training set.
transform = T.Compose([
    T.RandomRotation(degrees=69),
    T.RandomHorizontalFlip(p=0.6),
    T.RandomResizedCrop(250),
    T.CenterCrop(size=(25, 25)),
    T.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0),
    T.Grayscale(num_output_channels=3),
    T.Pad(padding=(2,3), fill=0, padding_mode='constant'),
    T.GaussianBlur(kernel_size=(9, 9)),
    T.RandomErasing(),
    T.ToTensor(),
    # Optionally add normalization
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = AnimalDataset(root=root, split="train", transform=transform)
valid_dataset = AnimalDataset(root=root, split="val", transform=test_transform)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")

# Model
model = resnet18.to(device)
print(model)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")


lr = 1e-3
gamma = 0.8
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)



lr = 1e-3
gamma = 0.8
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

# Train the model for 10 epochs.
# You should get an accuracy about 50%.
# Train the model for 5 epochs.
# You should get an accuracy about 50%.
n_epoch = 20
train(model, n_epoch, optimizer, scheduler)

def inference(model, data_loader, output_fn="predictions.txt"):
    """Generate predicted labels for the test set."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in tqdm(data_loader):
            images = images.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    with open(output_fn, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {output_fn}")
    return predictions


test_dataset = AnimalDataset(root=root, split="test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
inference(model, test_loader, "pred_resnet_ft.txt")