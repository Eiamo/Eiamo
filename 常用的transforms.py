from PIL import Image
from torchvision import  transforms
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")
img=Image.open(r"dataset/hymenoptera_data/train/ants/0013035.jpg")

trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("totensor",img_tensor)

print(img_tensor[0][0][0])

trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

img_norm=trans_norm(img_tensor)
writer.add_image("normalize1",img_norm)
print(img_norm[0][0][0])

writer.close()