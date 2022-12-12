import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_ssl(dataloader, model, loss_fn, optimizer):
    """
    Mini-batched training
    
    Args:
        dataloader ()     : 
        model (nn.Module) : 
        loss_fn ()        : loss/objective function
        optimizer ()      : optimization algorithm
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        # Compute prediction error
        (n, c, d1, d2) = X.shape
        X = torch.reshape(X, (-1, c*d1*d2))
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}]/{size:>5d}")
            
def test_ssl(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    accuracy = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            (n, c, d1, d2) = X.shape
            X = torch.reshape(X, (-1, c*d1*d2))
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy += 100*correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy

def train_vae(dataloader, model, loss_fn, optimizer, beta=1):
    """
    Mini-batched training
    
    Args:
        dataloader ()     : 
        model (nn.Module) : 
        loss_fn ()        : loss/objective function
        optimizer ()      : optimization algorithm
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(DEVICE)
        # Compute prediction error
        
        (n, c, d1, d2) = X.shape
        X = torch.reshape(X, (-1, c*d1*d2))
        Xrec, Mu, Logvar = model(X)
        loss = loss_fn(Xrec, X, Mu, Logvar, beta=beta)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}]/{size:>5d}")
            
def test_vae(dataloader, model, loss_fn, beta=1):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(DEVICE)
            
            (n, c, d1, d2) = X.shape
            X = torch.reshape(X, (-1, c*d1*d2))
            Xrec, Mu, Logvar = model(X)
            
            test_loss += loss_fn(Xrec, X, Mu, Logvar, beta=beta).item()
    
    test_loss /= num_batches
    
    return test_loss



"""
Loss Functions
"""

def vae_loss(x_hat, x, mu, logvar, beta=1):
    
    rec_loss = nn.functional.mse_loss(x, x_hat, reduction="sum")
    
    dkl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
    
    return rec_loss + beta * dkl

def train_uns(dataloader, model, loss_fn, optimizer):
    """
    Mini-batched training
    
    Args:
        dataloader ()     : 
        model (nn.Module) : 
        loss_fn ()        : loss/objective function
        optimizer ()      : optimization algorithm
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(DEVICE)
        # Compute prediction error
        
        (n, c, d1, d2) = X.shape
        X = torch.reshape(X, (-1, c*d1*d2))
        # print("tu train_uns, X shape:", X.shape)
        Xrec = model(X)
        # print("tu train_uns, Xrec shape:", Xrec.shape)
        loss = loss_fn(Xrec, X)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}]/{size:>5d}")
            
def test_uns(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(DEVICE)
            
            (n, c, d1, d2) = X.shape
            X = torch.reshape(X, (-1, c*d1*d2))
            Xrec = model(X)
            
            test_loss += loss_fn(Xrec, X).item()
    
    test_loss /= num_batches
    
    return test_loss