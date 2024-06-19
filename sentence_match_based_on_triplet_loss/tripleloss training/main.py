import torch
import os
import random
import numpy as np
from loader import load_data
from model import SiameseNetwork,choose_optimizer
from evaluate import Evaluator
from config import Config
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)


"""
模型训练主程序
"""
def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data=load_data(config["train_data_path"],config)
    #加载模型
    model=SiameseNetwork(config)
    #标识是否使用gpu
    cuda_flag=torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model=model.cuda()
    #加载优化器
    optimizer=choose_optimizer(config,model)
    #加载效果测试类
    evaluator=Evaluator(config,model,logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch+=1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss=[]
        for index,batch_data in enumerate(train_data):
            optimizer.zero_grad()  #梯度置零，用于每个batch梯度更新
            if cuda_flag:
                batch_data=[d.cuda() for d in batch_data]
            input_id1,input_id2,labels=batch_data
            loss=model(input_id1,input_id2,labels)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss:%f"% np.mean(train_loss))
        evaluator.eval(epoch)
    model_path=os.path.join(config["model_path"],"epoch_%d.pth"% epoch)
    torch.save(model.state_dict(),model_path)
    return



if __name__ == '__main__':
    main(Config)