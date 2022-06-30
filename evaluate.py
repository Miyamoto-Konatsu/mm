
import scipy.io
import torch
import numpy as np


# 测试准确率
def evaluate(qf,ql,gf,gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()

    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    my_answer = gl[index]
    list_top_K = [1, 2, 4, 8]
    ans = [0, 0, 0, 0]
    for i_ in range(len(list_top_K)):
        current_k = list_top_K[i_]
        if ql in my_answer[0:current_k]:
            ans[i_] = 1
    # dict_ = dict()
    # cnt = 8
    # image_=['','','','','','','','']
    # for answer in my_answer:
    #     if cnt == 0:
    #         break
    #     if answer not in dict_:
    #         dict_[answer] = -1
    #     if ql == answer:
    #         dict_[answer] += 1
    #     cnt -= 1
    #     images_ = os.listdir("./data_test/images/gallery/0"+str(answer))
    #     image_[7-cnt] = "./data_test/images/gallery/0"+str(answer)+'/'+images_[dict_[answer]]
    # imgs_path = "./data_test/images/query/0"+str(ql)+"/0.jpg"
    # img=cv2.imread(imgs_path)
    # plt.subplot(2, 8, 4)
    # plt.imshow(img)
    # cnt = 9
    # for im in image_:
    #     img = cv2.imread(im)
    #     plt.subplot(2 , 8 , cnt)
    #     plt.imshow(img)
    #     cnt += 1
    # plt.show()
    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index1 = np.argwhere(gl==-1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index = np.append(junk_index2, junk_index1) #.flatten())
    return np.array(ans)

result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])

query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])

gallery_label = result['gallery_label'][0]


query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
acc = np.array([0, 0, 0, 0])
#print(query_label)
for i in range(len(query_label)):
    this_feature = query_feature[i]
    ans = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
    acc += ans

acc = acc / len(query_label)
print(acc)


