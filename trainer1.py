import NETA
from NETA import netA
import NETC1
from NETC1 import netC
import torch.nn as nn
import torch.optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import netA1
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import model_selector
import parameters_calculator
from tqdm import tqdm

class trainer_class(nn.Module):
    def __init__(self,train_loader,validation_loader,args):
        super().__init__()
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        
        self.args=args


    def train(self):
        count=0
        ##do not write outer loop from here 
        val_loader=iter(self.validation_loader)
        neta=netA1.netA()
        n_parameters=parameters_calculator.count_parameters(neta)
        print(n_parameters)
        opt_neta=torch.optim.Adam(neta.fc_loc.parameters(),lr=0.0001)

        ##need to get netc_obj object here
        ##tell name of model to helper.get_netc_model()
        netc_obj=model_selector.get_netc_model("custom")
        opt_netc=torch.optim.Adam(netc_obj.parameters(),lr=0.001)
        loss_obj=nn.CrossEntropyLoss()
        count=0
        count2=0
        for epoch in range(10):
            val_loader=iter(self.validation_loader)
            tr_loader=iter(self.train_loader)
            for i in range(len(self.train_loader)):
                
                ##define outer loop from here 

                ###get images from train loader
                images,labels=next(tr_loader)

                
                
                ##create an object of netA and send image batch
                

                ##defining an optimizer for netA
                


                rot_imgs,rot_labels,affine_matrix,inverse_matrix=neta.transform_images(images,labels)
                
                
                ##^ obtained the rotated images from netA
                ##now we have both images and rot_imgs
                
                ##object of netc_obj
                

                ##passing rotated images through netc_obj

                output=netc_obj.forward(rot_imgs)
                
                
                ##defining loss
                
                loss=loss_obj(output,labels)
                
                ##gradients before backward for neta
                #for param in neta.fc_loc.parameters():
                #    print(param.data)

                ##gradients before backward for netc_obj
                #for param in netc_obj.parameters():
                #   print(param.data)
                
                #print(neta.fc_loc[0].weight.grad)
                #print(netc_obj.conv1.weight.grad)


                loss.backward(retain_graph=True)
                

                #print(neta.fc_loc[0].weight.grad)
                ##UPDATE only netc_obj
                opt_netc.step()

                opt_netc.zero_grad()
                opt_neta.zero_grad()    



                ##print(netc_obj.conv1.weight.grad)


                
                ##get the next from the validation batch
                val_imgs,val_labels=val_loader.next()
                
                
                

                    
                    
                
                
                # apply transformation +ve
                grids1 = F.affine_grid(affine_matrix[0:len(val_imgs),:,:], val_imgs.size(), align_corners=True)
                rot_val_imgs = F.grid_sample(val_imgs,grids1, align_corners=True)

                
                    
                

                ##apply inverse transformation
                grids2 = F.affine_grid(inverse_matrix[0:len(val_imgs),:,:], val_imgs.size(), align_corners=True)
                rot_val_imgs_inv = F.grid_sample(rot_val_imgs,grids2, align_corners=True)

                if i%100==0:
                    img1=val_imgs[0][0,:,:].numpy()
                    img2=rot_val_imgs[0][0,:,:].detach().numpy()
                    img3=rot_val_imgs_inv[0][0,:,:].detach().numpy()
                    
                    plt.figure()

                #subplot(r,c) provide the no. of rows and columns
                    f, axarr = plt.subplots(3,1) 

                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                    axarr[0].imshow(img1)
                    axarr[1].imshow(img2)
                    axarr[2].imshow(img3)
                    plt.show()
                                
                ##passing the untransformed(transformed) images through netc_obj
                outputs2=netc_obj.forward(rot_val_imgs_inv)
                
                m=0
                for i in outputs2:
                    x=torch.max(i)
                    c=((i == x).nonzero(as_tuple=True)[0])
                    if val_labels[m]==c:
                        count+=1
                    count2+=1
                    m+=1

                    
            
                

                
                    

                

                
                    
                val_loss=loss_obj(outputs2,val_labels)
                
                ##zeroing the grad from previous steps 
                opt_neta.zero_grad()
                opt_netc.zero_grad()

                #backpropogating with validation loss 
                val_loss.backward(retain_graph=True)
                

                ##optimizing only neta
                opt_neta.step()
            

                
                
                

                ##neta gradients after backward
                #for param in neta.fc_loc.parameters():
                #    print(param.data)
                
                ##netc_obj gradients after backward

                #for param in netc_obj.parameters():
                #   print(param.data)
                
                #rots_images=rot_imgs[5,0,:,:]
                #rots_images=rots_images.detach().numpy()
                
                #rots_images=rot_imgs[0,0,:,:]
                #rots_images=rots_images.detach().numpy()
                #print(rots_images.shape)
                print("accuracy={}   val_loss== {}  epoch=={}".format((count/count2*100),(val_loss),epoch))


          
            

                ###thing is i need to use the same grid with same netA 
                
            
                
            
            ###pass them to get the noise and the transformed images
            ## pass the transformed and regular images to the netc_obj
            ##use the loss to only back propogate through the netc_obj and not neta
            
