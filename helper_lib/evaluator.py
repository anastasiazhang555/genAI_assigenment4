import torch
def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)     
    model.eval()         

    total_loss = 0.0
    correct = 0
    total = 0

    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # get class index with max logit
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    # Calculate averages
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
