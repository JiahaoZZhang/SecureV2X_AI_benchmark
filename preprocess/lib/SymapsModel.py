from torch import nn, optim
import torch
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Convolution AutoEncoder Model Example 
class ConvAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4,
                      out_channels=64,
                      kernel_size=10,
                      padding=0,
                      stride=1),
            # nn.ReLU(True),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            
            nn.Conv1d(in_channels=64,
                      out_channels=32,
                      kernel_size=10,
                      padding=0,
                      stride=1),
            # nn.ReLU(True),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            
            # nn.Conv1d(in_channels=32,
            #     out_channels=16,
            #     kernel_size=10,
            #     padding=0,
            #     stride=1),
            # # nn.ReLU(True),
            # nn.LeakyReLU(),
            # # nn.Tanh(),
            # nn.Dropout(p=0.2),
        )
        
        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(in_channels=16,
            #           out_channels=32,
            #           kernel_size=10,
            #           stride=1),
            # # nn.ReLU(True),
            # nn.LeakyReLU(),
            # # nn.Tanh(),
            # nn.Dropout(p=0.2),
            
            
            nn.ConvTranspose1d(in_channels=32,
                      out_channels=64,
                      kernel_size=10,
                      stride=1),
            # nn.ReLU(True),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            
            
            nn.ConvTranspose1d(in_channels=64,
                      out_channels=4,
                      kernel_size=10,
                      stride=1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        






# Convolution Variationnel AutoEncoder Model Example 

class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        # print(inputs.shape)
        # print(self.module(inputs).shape)
        # print(self.module)
        return self.module(inputs) + inputs

class ConvVAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            
            nn.Conv1d(in_channels=4,
                out_channels=64,
                kernel_size=10,
                padding=0,
                stride=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            
            nn.Conv1d(in_channels=64,
                out_channels=128,
                kernel_size=10,
                padding=0,
                stride=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            
            
            ResNet(
                nn.Sequential(
                    nn.Conv1d(in_channels=128,
                            out_channels=32,
                            kernel_size=3,
                            padding=1,
                            stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2),
                    
                    nn.Conv1d(in_channels=32,
                            out_channels=128,
                            kernel_size=3,
                            padding=1,
                            stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2),
                )
            ),
            
            ResNet(
                nn.Sequential(
                    nn.Conv1d(in_channels=128,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2),
                    
                    nn.Conv1d(in_channels=32,
                    out_channels=128,
                    kernel_size=3,
                    padding=1,
                    stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2)
                )
            ),
            
            nn.Conv1d(in_channels=128,
            out_channels=8,
            kernel_size=10,
            padding=0,
            stride=1),
            # nn.ReLU(True),
            nn.LeakyReLU(),
            # nn.Tanh(),
        )
       
       
        # latent mean and variance 
        self.mean_layer = nn.Linear(23, 4)
        self.logvar_layer = nn.Linear(23, 4)
       
        
        self.decoder = nn.Sequential(
            
            nn.Linear(4, 23),
            nn.LeakyReLU(),
            # nn.Tanh(),
            
            nn.ConvTranspose1d(in_channels=8,
                out_channels=128,
                kernel_size=10,
                stride=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            
            ResNet(
                nn.Sequential(
                    
                    nn.Conv1d(in_channels=128,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2),         
                    
                    nn.Conv1d(in_channels=32,
                            out_channels=128,
                            kernel_size=3,
                            padding=1,
                            stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2),
                )
            ),
            
            ResNet(
                nn.Sequential(
                    nn.Conv1d(in_channels=128,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2),
                
                    nn.Conv1d(in_channels=32,
                        out_channels=128,
                        kernel_size=3,
                        padding=1,
                        stride=1),
                    # nn.ReLU(True),
                    nn.LeakyReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(p=0.2),
                )          
            ),
            
            nn.ConvTranspose1d(in_channels=128,
                out_channels=64,
                kernel_size=10,
                stride=1), 
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            
            nn.ConvTranspose1d(in_channels=64,
                out_channels=4,
                kernel_size=10,
                stride=1)
        )
        
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)    
        z = mean + var*epsilon
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        # print(x_hat.shape)
        return x_hat, mean, logvar




# 3D/2D convolution layer
class LstmAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4,
                      out_channels=64,
                      kernel_size=10,
                      padding=0,
                      stride=1),
            # nn.ReLU(True),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            
            nn.Conv1d(in_channels=64,
                      out_channels=32,
                      kernel_size=10,
                      padding=0,
                      stride=1),
            # nn.ReLU(True),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32,
                      out_channels=64,
                      kernel_size=10,
                      stride=1),
            # nn.ReLU(True),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            
            
            nn.ConvTranspose1d(in_channels=64,
                      out_channels=4,
                      kernel_size=10,
                      stride=1),
            nn.Dropout(p=0.2),
            
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        



# LSTM AE example



# LSTM VAE exampe



# CNN - LSTM



# Transformer ?




        
if __name__ == 'main':
    
    # example 
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True,
                            download=True, transform=tensor_transform)
    loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=32, shuffle=True)
    
    
    
    myModel = ConvAEModel()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(myModel.parameters(), lr=1e-3, weight_decay=1e-8)
    
    
    epochs = 20
    outputs = []
    losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    myModel.to(device)

    for epoch in range(epochs):
        for images, _ in loader:
            images = images.view(-1, 28 * 28).to(device)
            
            reconstructed = myModel(images)
            loss = loss_function(reconstructed, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        outputs.append((epoch, images, reconstructed))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    
    