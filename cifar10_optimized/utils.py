#everything I'll need but don't want to write over and over again
import torch 
import matplotlib.pyplot as plt

#Early Stopping Boilerplate
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
                # kill it 
                self.early_stop=True
        else:
            #here is the reset. 
            self.best_loss = val_loss
            self.counter = 0 

# Basic Classification Training Loop
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

#Basic Classification Testing Loop
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

#normalize function 
def pytorch_normalize(X_tensor):
    mean = X_tensor.mean(dim=0, keepdim=True)
    std = X_tensor.std(dim=0, keepdim=True)
    return (X_tensor - mean) / (std + 1e-9) #to prevent dividing by 0

# FUNCTION TO PLOT LOSSES
def plot_losses(train_losses,val_losses,save_path,session=1):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train/Val Losses S{session}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()