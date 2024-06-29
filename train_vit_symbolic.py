import torch
import torchvision
from torchvision.transforms import v2   
from transformers import ViTModel, ViTConfig
from einops import rearrange
from kan import KAN

config = ViTConfig.from_pretrained("mrm8488/vit-base-patch16-224-pretrained-cifar10")
print(config.hidden_size)
model = ViTModel(config=config, add_pooling_layer=False).to("cuda")

transform = v2.Compose(
    [v2.ToTensor(),
     v2.Resize((224, 224)),
     v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 2

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

dataset = {}
symbolic_kan = KAN(width=[768,100,1], grid=5, k=3)

for x, y in trainloader:
    test = model.forward(x.to("cuda"))
    print(test.last_hidden_state[:,0,:].shape)
    y = rearrange(y, "b -> b 1")
    print(y.shape)
    dataset["train_input"] = test.last_hidden_state[:, 0, :].to("cpu")
    dataset["train_label"] = y
    break

for x, y in testloader:
    test = model.forward(x.to("cuda"))
    print(test.last_hidden_state[:, 0, :].shape)
    y = rearrange(y, "b -> b 1")
    print(y.shape)
    dataset["test_input"] = test.last_hidden_state[:, 0, :].to("cpu")
    dataset["test_label"] = y
    break

symbolic_kan.train(dataset=dataset)

    
