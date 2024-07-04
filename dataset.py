from utils import *

class CelebaData(Dataset):
    def __init__(self):
        self.path='GAN/celeba/img_align_celeba/img_align_celeba'
        self.files=os.listdir(self.path)
        self.transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f=self.files[idx]
        fpath=os.path.join(self.path, f)
        im=cv2.imread(fpath)
        im=Image.fromarray(im)
        im=self.transform(img=im)
        return im
