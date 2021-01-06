import torch
import numpy as np
import time


def train(epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.Inf
    start_time = time.time()

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        # TRAINING
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        # VALIDATION
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            val_output = model(data)
            val_loss = criterion(val_output, target)
            valid_loss = ((1 / (batch_idx + 1)) * (val_loss.data - valid_loss))

        # Print Training and Validation Statistics
        print('Epoch: {}/{} \t\tTime taken: {:.6f} seconds'.format(
            epoch,
            epochs,
            time.time()-epoch_start_time
        ))
        print('------------------------------------------------------------------')
        print('Training Loss: {:.6f}    Validation Loss: {:.6f}\n'.format(
            train_loss,
            valid_loss
        ))

        # Save Model IF Validation Loss is Decreasing
        if(valid_loss < valid_loss_min):
            print('-------------------------  SAVING MODEL  -------------------------')
            print('Old Validation Loss: {:.6f}   >>>>>   New Validation Loss: {:.6f}\n'.format(
                valid_loss_min,
                valid_loss_min
            ))
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)

    print('\nTotal Training Time: {:.6f} seconds'.format(
        time.time()-start_time))

    return model
