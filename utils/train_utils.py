import os
import torch
import datetime

# a function for test loss and validation pass


def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # forward pass
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # calculating accuracy
        # model return is log-softmax, take exponential to get the probabilities
        # class with hihgest prob. iis our predicted class, compare to true label
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])

        # accuracy is the number of correct predictions divided by all prediction (take the mean)
        # Use .float() for device-agnostic conversion
        accuracy += equality.float().mean()

    return test_loss, accuracy


# function to training the model

def train_model(model, trainloader, validloader, criterion, optimizer, device,
                 arch, save_dir, classifier, dataset, epochs=10, print_every=40):
    '''
    Arguments: The model, dataset of trainloader and validloader, criterion, the optimizer, choice of gpu power or cpu, the number of epochs,
    Returns: Nothing
    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" 
    step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively
    '''

    print('<---- Starting Neural Network Training ---->')
    print(f'<---- Model Training On {device}---->')
    steps = 0
    running_loss = 0
    model.to(device)

    best_valid_loss = float('inf')

    for e in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            # setting gradients to zero
            optimizer.zero_grad()

            # forward and backward pass
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()

            # updating weights
            optimizer.step()

            running_loss += loss.item()
            torch.cuda.empty_cache()

            if steps % print_every == 0:
                # turns out dropout mode in inference
                model.eval()

                # turns off gradient for validation, saves memory and computation
                with torch.no_grad():
                    valid_loss, accuracy = validation(
                        model, validloader, criterion, device)

                avg_valid_loss = valid_loss/len(validloader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(avg_valid_loss),
                      "Accuracy: {:.3f}".format(accuracy/len(validloader)))

                if avg_valid_loss < best_valid_loss:
                    print("Validation loss decreased ({:.3f} --> {:.3f}). Saving model...".format(
                        best_valid_loss, avg_valid_loss))
                    best_valid_loss = avg_valid_loss
                    save_checkpoint(model, arch, save_dir,
                                    classifier, optimizer, dataset, epochs)

                running_loss = 0
                model.train()
            torch.cuda.empty_cache()


def save_checkpoint(model, arch, save_dir, classifier, optimizer, train_datasets, epochs):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model at a specified by the user path
    '''
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {
        'model': model,
        'arch': arch,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }

    # check if save_dir exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Modified path
    path = f'{save_dir}/{arch}_model_checkpoint_{epochs}ep.pth'
    torch.save(checkpoint, path)

    print('<---- Checkpoint saved to: {}---->'.format(path))