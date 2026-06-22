# SECTION 1 - necessary imports ############################################
import torch 
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import json
from utils import train_loop_class, test_loop, EarlyStopping, plot_losses

# DEVICE SETUP
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# SECTION 2 - DATASETS AND DATA LOADERS ############################################

# TRANSFORMS
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

#DATASET
full_data = datasets.CIFAR10(root="./data_cifar10",train=True,transform=test_transforms,download=False)

main_num_classes = len(full_data.classes)
main_num_channels = full_data[0][0].shape[0]

train_size = int(0.8*len(full_data))
val_size = len(full_data)-train_size
split_lengths = [train_size,val_size]
train_dataset, val_dataset = random_split(full_data,split_lengths)

train_dataset.dataset.transform = train_transforms

test_dataset = datasets.CIFAR10(root="./data_cifar10",train=False,transform=test_transforms,download=False)

#DATALOADERS
batch_size = 32
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# SECTION 3 - MODEL RELOADING AND UNFREEZING ############################################

#WE need to repeat the exact freezing and unfreezing before we do more. It's good practice

#1. Load pre-trained model 
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

#2. Freeze all layers
for param in model.parameters():
    param.requires_grad=False

#2.a The exact unfreezing you did before, but not yet what you want to do now
for param in model.layer4.parameters():
    param.requires_grad=True
for param in model.layer3.parameters():
    param.requires_grad=True

#3. Clip output layer to your needs
orig_num_feats = model.fc.in_features
model.fc = nn.Linear(orig_num_feats, main_num_classes)

#4.a LOAD SAVED MODEL WEIGHTS, otherwise you're toast
model_save_path_s2 = "s2_best_model.pth"
model.load_state_dict(torch.load(model_save_path_s2,weights_only=True))
model = model.to(device)

#4.b The unfreezing you want to do now
for param in model.layer2.parameters():
    param.requires_grad=True
for param in model.layer1.parameters():
    param.requires_grad=True

#5. Updated optimizer
optimizer_s3 = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-7)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
lr_scheduler_s3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=100, eta_min=1e-7)

# SECTION 4 - RETRAINING FOR SESSION 3

#previous val loss and best test loss to beat 
with open("./results/training_results.json","r") as f:
    previous_results = json.load(f) #and then it's just a dict from here

best_val_loss_s2 = previous_results["best_val_loss_s2"]
curr_best_test_loss = previous_results["test_loss"]
print(f"Continuing from Stage 2 best val loss: {best_val_loss_s2}")
print(f"Previous test loss: {curr_best_test_loss}")

train_losses_d2 = []
val_losses_d2 = []
model_save_path_s3 = "s3_best_model.pth"
best_loss_s3 = train_loop_class(model,device,optimizer_s3,loss_fn,lr_scheduler_s3,train_dataloader,val_dataloader,train_losses_d2,val_losses_d2,model_save_path_s3,EarlyStopping(6,0.01),epochs=20,initial_loss=best_val_loss_s2)

#going forward, save results to json without rounding for better precision
#save to json
#save to json
with open("results/training_results.json", "r") as f:
    results = json.load(f)

results["best_val_loss_s3"] = best_loss_s3
results["stage"] = "layer2_layer1_unfrozen"

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved. Day 2. Session 3")

# SECTION 5 - Testing and Deploying 
test_loss, test_accuracy = test_loop(model, device, loss_fn, test_dataloader)
print(f"S3 Test Loss: {test_loss:.4f} | Accuracy: {test_accuracy*100:.2f}%")

#Update json 
with open("results/training_results.json","r") as f:
    results = json.load(f)

results["test_loss_s3"] = test_loss
results["test_accuracy_s3"] = test_accuracy

with open("results/training_results.json","w") as f:
    json.dump(results,f,indent=2)

# Plot and save 
plot_losses(train_losses_d2,val_losses_d2,"results/losses_s3.png",session=3)


