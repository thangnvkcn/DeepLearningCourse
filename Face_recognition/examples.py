from image_batch_generator import image_batch_generator
from loss_function import loss_tensor
from model import train_net
X=[]
y=[]
net = train_net((221, 221, 3))
model  = net.deep_rank_model()
model.fit_generator(generator = image_batch_generator(X, y, batch_size=24), steps_per_epoch=len(X)//24, epochs=2000, verbose=1)