# -*- coding:utf-8 -*-
import re
#from dataloder_v2 import *

from CMP_model import *  #v2:add BatchNorm layer
from multiprocessing import cpu_count
from model_utils import *
from mxnet.gluon import utils as gutils
from dataloder_v4_big_label import *

# def train():
#plt.ion()
epoch_loss = {}
BATCH_SIZE = 32
num_epochs = 200
learning_rate = 1e-4
CTX = [mx.gpu(2), mx.gpu(3)]
FREEZE_TOP_LAYERS = 12
train_data_root = './synth_dataset/'
TAG = 'huge_label'
viz_op_dir = 'viz_op_stages/'+TAG
if not os.path.exists(viz_op_dir):
    os.mkdir(viz_op_dir)

VISUALIZE = True
train_dataset = ImageWithMaskDataset(root=train_data_root)
train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_count()) #cpu_count()

mask0 = create_mask(BATCH_SIZE,CTX[0])
mask1 = create_mask(BATCH_SIZE,CTX[1])

model = CPM(stages=6,joints=21)
model.hybridize()
print('loading parameters and initializing model......')#
model.load_params('./model_saved/big_label_148.params',ctx=CTX)
#model.load_params('./model_saved/init_bn.params',ctx=CTX)
if FREEZE_TOP_LAYERS != 0:
    for layer_name, layer_params in model.collect_params().items():
        if 'sub_stage' in layer_name :
            if int(re.findall(r"\d",layer_name)[-1]) < FREEZE_TOP_LAYERS:
                layer_params.grad_req='null'

#model.collect_params().initialize(mx.init.Xavier('gaussian'), ctx=CTX)
#model.load_params('./model_initial/cpm_initial.params', ctx=CTX)

#criterion = g.loss.L2Loss()
optimizer = g.Trainer(model.collect_params(), 'adam',{'learning_rate': learning_rate, 'wd': 1e-5})

for epoch in range(num_epochs):
    loss_list = []
    num_iters=0
    for img, label in train_data_loader:
        num_iters += 1
        gpu_imgs = gutils.split_and_load(img, CTX)
        gpu_labels = gutils.split_and_load(label, CTX)

        gpu_labels[0] = gpu_labels[0] * mask0
        gpu_labels[1] = gpu_labels[1] * mask1
        #batch = img.shape[0]
        # ===================forward=====================
        with mx.autograd.record():
            outputs = [model(img) for img in gpu_imgs]
            outputs[0] = outputs[0] * mask0
            outputs[1] = outputs[1] * mask1
            loss = [my_l2_loss(output, label) for output, label in zip(outputs, gpu_labels)]
    # ===================backward====================
        if VISUALIZE:
            if num_iters%200==0:
                print('Starting Visualizing.......')
                save_stage_output(gpu_imgs,outputs, gpu_labels, num_iters,epoch,save_dir=viz_op_dir)
        for l in loss:
            l.backward()
        optimizer.step(BATCH_SIZE)
        mx.nd.waitall()
        print ('Batch AVG loss is: %f ----- in epoch %d' % ((sum(loss)[0].asscalar())/len(loss),epoch + 1))
        loss_list.append((sum(loss).asscalar())/len(loss))
    if ((epoch + 1) % 3 == 0 and (epoch + 1) != num_epochs):
        model.save_params('./model_saved/'+TAG+'_%d.params' % (epoch + 1))
        pickle_NDArray({'gpu_labels': gpu_labels, 'outputs': outputs,'img':gpu_imgs}, TAG+'_dct%d.pkl' %(epoch+1))


    # ===================log========================
    epoch_mean_loss = np.mean(loss_list)

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_mean_loss))
    epoch_loss['epoch_%d' % (epoch + 1)] = float(epoch_mean_loss)

model.save_params('./model_saved/final_sigma_2.params')
with open('./model_saved/loss_log.json', 'w') as js_f:
    js_f.write(json.dumps(epoch_loss))