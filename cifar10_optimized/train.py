# SECTION 1 - necessary imports ############################################
import torch 
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import json

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
full_data = datasets.CIFAR10(root="./data_cifar10",train=True,transform=test_transforms,download=False) # train transform applied to train split below, after random_split

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

# SECTION 3 - MODEL INSTANTIATION ############################################

#1. load model
model = models.resnet18(models.ResNet18_Weights.DEFAULT)

#2. freeze all layers 
for param in model.parameters():
    param.requires_grad = False

#3. clip output layer to our needs 
orig_num_feats = model.fc.in_features 
model.fc = nn.Linear(orig_num_feats,main_num_classes)

#4. if that is how it played out, then do optimizer
model = model.to(device)
optimizer = optim.Adam(model.fc.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-6)

# SECTION 4 - MODEL TRAINING ############################################

# Early Stopping class to prevent overfit
class EarlyStopping:

    def __init__(self,patience=5,min_delta=0):
        self.patience = patience #how many epochs to wait after last improvement 
        self.min_delta = min_delta #minimum change to qualify as improvement
        self.counter = 0 
        self.best_loss = None
        self.early_stop = False

    def __call__(self,val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop=True
        else:
            #here is the reset. 
            self.best_loss = val_loss
            self.counter = 0 

#Training Loop

def train_loop_class(model, device, optimizer, loss_fn, lr_scheduler, train_dataloader, val_dataloader, train_losses, val_losses, model_save_path, early_stopping, epochs=100, initial_loss=float("inf")):

    best_loss = initial_loss

    for epoch in range(epochs):
        # == TRAINING ==
        model.train()
        train_loss = 0.0

        for X_train, y_train in train_dataloader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        if ((epoch + 1) == 1) or ((epoch + 1) % 10 == 0):
            print(f"Epoch {epoch+1} ; Train Loss: {avg_train_loss:.4f}")
        
        # == VAL ==
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                y_pred = model(X_val)
                loss = loss_fn(y_pred, y_val)
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        if ((epoch + 1) == 1) or ((epoch + 1) % 10 == 0):
            print(f"Epoch {epoch+1} ; Val Loss: {avg_val_loss:.4f}")
        
        # == LR Scheduling ==
        lr_scheduler.step()

        # == SAVE BEST MODEL ==
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)

        # == EarlyStopping ==
        if early_stopping is not None:  # just to be safe
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_loss

# plotting losses function

def plot_results(train_losses,val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

####### TRAINING CALL #####

train_losses = []
val_losses = []
model_save_path_s1 = "s1_best_model.pth"
best_loss_s1 = train_loop_class(model, device, optimizer, loss_fn, lr_scheduler, train_dataloader, val_dataloader, train_losses, val_losses, model_save_path_s1, EarlyStopping(5,0.01), epochs=100, initial_loss=float("inf"))

#save to json
results = {
    "model": "ResNet18",
    "dataset": "CIFAR-10",
    "num_classes": main_num_classes,
    "best_val_loss_s1": round(best_loss_s1, 4),
    "stage": "classifier_only"
}

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Results saved for training session 1!")

# SECTION 5 - UNFREEZING AND RE TRAINING ############################################

#WE need to repeat the exact freezing and unfreezing before we do more. It's good practice

#1. Load pre-trained model 
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

#2. Freeze all layers
for param in model.parameters():
    param.requires_grad=False

#2.a The exact unfreezing you did before, but not yet what you want to do now

#3. Clip output layer to your needs
orig_num_feats = model.fc.in_features
model.fc = nn.Linear(orig_num_feats, main_num_classes)

#4.a LOAD SAVED MODEL WEIGHTS, otherwise you're toast
model.load_state_dict(torch.load(model_save_path_s1,weights_only=True))

#4.b The unfreezing you want to do now
for param in model.layer4.parameters():
    param.requires_grad=True
for param in model.layer3.parameters():
    param.requires_grad=True

#5. Updated optimizer
optimizer_s2 = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
lr_scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=100, eta_min=1e-6)

model_save_path_s2 = "s2_best_model.pth"
model = model.to(device)
loss_fn = loss_fn.to(device)
best_loss_s2 = train_loop_class(model, device, optimizer_s2, loss_fn, lr_scheduler_s2, train_dataloader, val_dataloader, train_losses, val_losses, model_save_path_s2, EarlyStopping(5,0.01), epochs=100, initial_loss=best_loss_s1)

#save to json
with open("results/training_results.json", "r") as f:
    results = json.load(f)

results["best_val_loss_s2"] = round(best_loss_s2, 4)
results["stage"] = "layer4_layer3_unfrozen"

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

# SECTION 6 - TESTING AND DEPLOYING ##############################

def test_loop(model,device,loss_fn,test_dataloader):

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X,y in test_dataloader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred,y)

            test_loss += loss.item()

            _,predicted = torch.max(y_pred,1)
            correct+= (predicted==y).sum().item()
            total += y.size(0)

    avg_test_loss = test_loss/len(test_dataloader)
    test_accuracy = correct/total

    return avg_test_loss,test_accuracy

test_loss, test_accuracy = test_loop(model, device, loss_fn, test_dataloader)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}%")

# Update JSON with test results
with open("results/training_results.json", "r") as f:
    results = json.load(f)

results["test_loss"] = round(test_loss, 4)
results["test_accuracy"] = round(test_accuracy * 100, 2)

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Final results saved!")