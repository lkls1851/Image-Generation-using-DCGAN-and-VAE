from utils import *
from dataset import CelebaData
from generator import Generator
from discriminator import Discriminator

ds=CelebaData()
dataloader=DataLoader(dataset=ds, batch_size=32, shuffle=True)
os.makedirs('Output', exist_ok=True)
num_epochs=100
nz=100
nc=3
nf=64
nd=64

modelG=Generator(nz=nz, nf=nf, nc=nc)
modelD=Discriminator(nc=nc, nd=nd)

device='cuda'

modelG.to(device)
modelD.to(device)

optimiserG=optim.Adam(modelG.parameters(), lr=1e-4)
optimiserD=optim.Adam(modelD.parameters(), lr=1e-4)

criterion=nn.BCELoss()

fixed_noise=torch.randn(64, nz, 1, 1, device=device)

real_label=1
fake_label=0

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        modelD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = modelD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = modelG(noise)
        label.fill_(fake_label)
        output = modelD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimiserD.step()
        modelG.zero_grad()
        label.fill_(real_label)  
        output = modelD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimiserG.step()

        if i % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 5 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = modelG(fixed_noise).detach().cpu()
                save_im=Image.fromarray(fake)
                save_name=str(iters)+'.jpg'
                save_im.save(os.path.join('Output', save_name))


        iters += 1
