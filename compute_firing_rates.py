from models import DSRN
import models
from helper import CustomMNIST
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt 



model = DSRN()

checkpoint_path = './models/checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print('Model loaded.')

model.eval()

mnist_test_path = './datasets/MNIST/raw/mnist_test.csv'
test_loader = torch.utils.data.DataLoader(
                CustomMNIST(csv_file=mnist_test_path, 
                        transform=transforms.Compose([
                            transforms.Normalize(0.1325, 0.3105)
                        ])), batch_size=1, shuffle=True)

sample = next(iter(test_loader))
output = model(sample[0])

# print(models.avg_firing_rates)
y = list(str(k) for k in models.avg_firing_rates.keys())
x = [v[1] for v in models.avg_firing_rates.values()]
plt.bar(y, x)
plt.title('Firing rates')
plt.xlabel('Layer')
plt.ylabel('Average firing rates over neuron of each layer')
plt.show()