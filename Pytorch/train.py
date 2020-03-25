import h5py
from util import generate_batch, create_net
import torch
torch.cuda.current_device()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epochs = 50
steps = 200
batch_size = 64


def main():
    model = create_net()

    for i in range(epochs):
        # iteratively add noise during training. Full noise after 66% of the set.
        noise_level = 3*i/epochs
        if noise_level >= 2: noise_level = 2
        #model = model.cuda()
        model = model.train().to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        for batch, labels in generate_batch(batch_size, noise_level):
            batch = batch.float().to(device)
            labels = labels.float().to(device)
            logit = model(batch)
            loss = criterion(logit, labels)
            print("doing batch, loss = " + str(loss.detach().cpu()))
            loss.backward()
            optimizer.step()



if __name__ == "__main__":
    main()
