import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os
import io
from PIL import Image

import preprocess_dataset
import cnn_model
import train
import test


print('\nBreast Cancer Detection! (Invasive Ductal Carcinoma!)')
print('=====================================================')
print('Invasive ductal carcinoma (IDC), sometimes called infiltrating \nductal carcinoma, is the most common type of breast cancer. \nAbout 80% of all breast cancers are invasive ductal carcinomas.\n')


use_cuda = torch.cuda.is_available()


def training_method():
    print('\nNew Model Training:')

    loaders = preprocess_dataset.preprocess()
    model = cnn_model.Net()

    if use_cuda:
        model = model.cuda()

    print('\nOur CNN Model Architecture:')
    print(model)

    epochs = int(input('\nEnter the number of epochs to train the model: '))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    trained_model_name = input(
        'Enter a filename for the trained model: ') + '.pt'
    if ' ' in trained_model_name:
        trained_model_name = trained_model_name.replace(' ', '_')

    print('\nTraining...')
    model = train.train(epochs, loaders, model, optimizer,
                        criterion, use_cuda, trained_model_name)

    print('\nNew model file saved: {}\n\n'.format(trained_model_name))



def testing_method():
    print('\nModel Testing:')

    while True:
        try:
            test_model = input('Enter saved model filename: ')
        except ValueError:
            print('Sorry, your input was is valid.')
            continue

        if not os.path.exists(test_model):
            print('Sorry, File does not exists.')
            continue
        else:
            break

    print('\nModel selected: {}'.format(test_model))

    loaders = preprocess_dataset.preprocess()
    criterion = nn.CrossEntropyLoss()

    model = cnn_model.Net()
    model.load_state_dict(torch.load(test_model))
    print('Model loaded\n\nTesting...')

    test.test(loaders, model, criterion, use_cuda)
    print('\n\n')



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize(50),
                                    transforms.CenterCrop(50),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [
                                                         0.229, 0.224, 0.225])
                                    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze_(0)



def get_prediction(image_tensor, model):
    output = model(image_tensor)
    _, pred = torch.max(output.data, 1)
    return pred



def input_testing_method():
    print('\nDetecting Breast Cancer from Input image:')

    while True:
        try:
            input_test_model = input('Enter saved model filename to use: ')
        except ValueError:
            print('Sorry, your input was is valid.')
            continue

        if not os.path.exists(input_test_model):
            print('Sorry, File does not exists.')
            continue
        else:
            break

    print('\nModel selected: {}'.format(input_test_model))

    model = cnn_model.Net()
    model.load_state_dict(torch.load(input_test_model))
    model.eval()
    print('Model loaded\n')
    print('Copy your 50*50 pixels histopatholy image patch to folder "test_inputs"')
    
    while True:
        try:
            image_file = './test_inputs/' + input('Enter image file name: ')
        except ValueError:
            print('Sorry, your input was is valid.')
            continue

        if not os.path.exists(image_file):
            print('Sorry, File does not exists.')
            continue
        else:
            if not allowed_file(image_file):
                print('Sorry, File format not supported.')
                continue
            
            break

    print('\nImage selected: {}'.format(image_file))
    with open(image_file, 'rb') as imageFile:
        img_bytes = imageFile.read()
    tensor = transform_image(img_bytes)
    prediction = get_prediction(tensor, model)
    result = prediction.item()

    print('\nResult: {}\n\n'.format(['Not Cancerous Cell', 'Cancerous Cell'][result]))



def program_flow():
    print('Input the number associated with the task you want to perform from the menu below and press Enter.')
    print('1) Train a new model \n2) Test the trained model with test dataset \n3) Detect Breast Cancer from input image \n4) Exit')

    while True:
        try:
            user_input = input('Please enter 1, 2, 3 or 4: ')
        except ValueError:
            print('Sorry, your input was is valid.')
            continue

        if not user_input in ['1', '2', '3', '4']:
            print('Sorry, your input was not valid. Please try again.')
            continue
        else:
            break

    if (user_input == '1'):
        training_method()
        program_flow()
    elif (user_input == '2'):
        testing_method()
        program_flow()
    elif (user_input == '3'):
        input_testing_method()
        program_flow()
    elif (user_input == '4'):
        print('\nBye!')
        return



# Initialize program
program_flow()
