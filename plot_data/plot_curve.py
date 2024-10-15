import datetime
import matplotlib.pyplot as plt
import random
import numpy as np

def plot_loss_and_lr(train_loss, Accuracy):
    train_loss = np.array(train_loss)
    Accuracy = np.array(Accuracy)
    try:
        
        x = list(range(len(train_loss[:,0])))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss[:,0], 'r', label='mass_loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and Accuracy")
    
        ax1.plot(x, train_loss[:,1], 'c', label='mass_score_loss')


        ax2 = ax1.twinx()
        ax2.plot(x, Accuracy[:,0],'b', label='Mass_Accuracy')
        ax2.plot(x, Accuracy[:,1],'k', label='RMSE_Accuracy')
        ax2.plot(x, Accuracy[:,2],'m', label='Mass_Score_Accuracy')
        ax2.set_ylabel("Accuracy")
        ax2.set_xlim(0, len(train_loss[:,0]))  # 设置横坐标整数间隔
        

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2 , 
                   labels1 + labels2 , loc='center right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('result_map_loss\\loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('result_map_loss\\mAP.png')
        plt.close()

        print("successful save mAP curve!")
    except Exception as e:
        print(e)


if __name__ == "__main__":

    file_path = 'result_map_loss\\mass_score_20_Conv_attention.txt'  # 将文件路径替换为实际的文件路径
    training_loss = []
    accuracy = []
    with open(file_path, 'r') as file:
    # 逐行读取文件内容
        for line in file:
            # 使用 split() 函数将每一行按空格分隔成单词列表
            words = line.split()
            training_loss.append(list(map(float,words[13:16])))
            accuracy.append(list(map(float,words[16:19])))
    # 读取完成后关闭文件
    file.close()
   
    plot_loss_and_lr(training_loss,accuracy)
    #plot_map(random_list2)