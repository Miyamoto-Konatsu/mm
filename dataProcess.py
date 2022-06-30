# -*-coding:utf-8-*-
import json
import os
import pickle
import cv2
import numpy
from tqdm import tqdm
# 用这个
#  image\image_text 里包含 train vla gallery query
def join_and_make(in_path1, in_path2=''):
    # 连接路径，如果不存在就新建
    if in_path2 == '':
        out_path = in_path1
    else:
        out_path = os.path.join(in_path1, in_path2)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    return out_path
# 说明
# 词向量的处理使冗余的，便于后期对路径进行替换得到对应的文件信息 image->image_text, .jpg->.txt
# 预处理流程目录

# 1.生成全语料库的词频表（商品库非视频库）

# 2.生成train与val

# 3.生成gallery与query

# 训练数据起止
train_start = 0
train_end = 500

# 测试数据起止
test_start = 500
test_end = 5000

# val或query 先放这些 且 只放这些 才放进其他（train或query）
image_val_or_query_num = 1
video_val_or_query_num = 1

# train或query 先放这些 且 只放这些 才放进其他（val或query）
image_train_or_gallery_num = 5
video_train_or_gallery_num = 15


# 原始数据集所在路径

target_path = join_and_make("./target")
###
image_path = join_and_make(target_path, 'image')
image_train_path = join_and_make(image_path, 'train')
image_val_path = join_and_make(image_path, 'val')
image_gallery_path = join_and_make(image_path, 'gallery')
image_query_path = join_and_make(image_path, 'query')

image_text_path = join_and_make(target_path, 'image_text')
image_text_train_path = join_and_make(image_text_path, 'train')
image_text_val_path = join_and_make(image_text_path, 'val')
image_text_gallery_path = join_and_make(image_text_path, 'gallery')
image_text_query_path = join_and_make(image_text_path, 'query')

####

source_path = r"F:\wab\disk7\leo.gb\TianchiDataset\train_dataset_7w\train_dataset_part1"

image_path = os.path.join(source_path, "image")
image_annotation_path = os.path.join(source_path, "image_annotation")
image_text_path = os.path.join(source_path, "image_text")

video_path = os.path.join(source_path, "video")
video_annotation_path = os.path.join(source_path, "video_annotation")
video_text_path = os.path.join(source_path, "video_text")
# </block>全局变量定义
tag_process = dict()        # 处理过的商品图片库/视频
image2image_val = dict()    # 图像存入验证集数量记录
video2image_val = dict()    # 视频关键帧存入验证集数量记录
image2image_query = dict()  # 图像存入查询集数量记录
video2image_query = dict()  # 视频关键帧存入查询集数量记录

def add_sectence_embedding(out_save_path, folder_id, num_files, image_or_video):
    if image_or_video == 'image':
        text_path = os.path.join(image_text_path, folder_id+'.txt')
    if image_or_video == 'video':
        text_path = os.path.join(video_text_path, folder_id+'.txt')
    save_path = out_save_path.replace('image', 'image_text')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, str(num_files) + '.txt')
    sentence_embedding = get_index_list(text_path)
    numpy.savetxt(save_file, sentence_embedding, fmt='%f')


def get_index_list(data):   # 生成句向量
    with open(data) as f:
        index_list_str = f.read()
        index_list = index_list_str.split(',')
        index_list = numpy.array(index_list)
        ret_dict = dict()  # rec_dic存储的是每个词在这个video_text中出现的次数
        sum_freq = 0  # 记录本商品图片 中 所有词在词频表中出现次数之和
        for item_index in index_list:
            if len(item_index) == 0:  # text文件为空
                vec_i = -1
                return numpy.zeros((100, 1))  # 返回100*1的0向量
            else:  # text文件非空
                vec_i = int(item_index)
                if vec_i in frequency_dict.keys():  # 词频表中有这个词才统计它的词频，否则当作无效词处理
                    if vec_i not in ret_dict:  # 该词第一次出现在句子中
                        ret_dict[vec_i] = 1
                    else:  # 不是第一次出现这个句子中
                        ret_dict[vec_i] += 1
                    sum_freq += frequency_dict[vec_i]
        flag = 1
        ans_list = []
        for vec_i in ret_dict.keys():
            vector_ = numpy.array(index2vec_dict[vec_i])
            weight: float = frequency_dict[vec_i] * ret_dict[vec_i] / sum_freq
            vector_ = [i * weight for i in vector_]
            if flag == 1:
                ans_list = numpy.array(vector_)
                flag = 0
            else:
                ans_list = ans_list + vector_
    if sum_freq == 0:
        return numpy.zeros((100, 1))
    return ans_list


def check(path):
    if path in tag_process:
        # 一组原始数据不能被重复标记为数据集
        # 输入为含有若干json文件的文件夹
        processed = Exception("此组文件已被重复添加到训练集/验证集/商品集/查询集")
        raise processed
    else:
        tag_process[path] = 1


def count(record, item):
    # 此item的商品存到val或query的计数
    if item in record:
        record[item] += 1
    else:
        record[item] = 1
    return record[item]

def generate_frame_image(video_path, frame_index, data_annotation, data_type, folder_id, choices=''):

    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_index)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    ret, frame = cap.read()
    if ret == True:
        for this_data in data_annotation:
            item_instance_id = this_data["instance_id"]
            if item_instance_id != 0:
                save_id = folder_id

                if data_type == 'gallery_all':
                    save_path = os.path.join(image_gallery_path, save_id)

                if data_type == 'query_all':
                    save_path = os.path.join(image_query_path, save_id)

                if data_type == 'train_all':
                    save_path = os.path.join(image_train_path, save_id)

                if data_type == 'val_all':
                    save_path = os.path.join(image_val_path, save_id)

                if data_type == 'gallery_first':
                    ret = count(image2image_query, save_id+'_video')
                    if ret <= image_train_or_gallery_num:
                        # gallery分配不足，优先分配gallery
                        save_path = os.path.join(image_gallery_path, save_id)
                    else:
                        # gallery分配充足，可以分配给query
                        save_path = os.path.join(image_query_path, save_id)

                if data_type == 'query_first':
                    ret = count(image2image_query, save_id+'_video')
                    if ret <= image_val_or_query_num:
                        save_path = os.path.join(image_query_path, save_id)
                        # query分配不足，优先分配query
                    else:
                        # query分配充足，可以分配给gallery
                        save_path = os.path.join(image_gallery_path, save_id)

                if data_type == 'train_first':
                    ret = count(image2image_val, save_id+'_video')
                    if ret <= image_train_or_gallery_num:
                        # train分配不足，优先分配train
                        save_path = os.path.join(image_train_path, save_id)
                    else:
                        # train分配充足，可以分配给val
                        save_path = os.path.join(image_val_path, save_id)

                if data_type == 'val_first':
                    ret = count(image2image_val, save_id+'_video')
                    if ret <= image_val_or_query_num:
                        # val分配不足，优先分配val
                        save_path = os.path.join(image_val_path, save_id)
                    else:
                        # val分配充足，可以分配给train
                        save_path = os.path.join(image_train_path, save_id)

                if data_type == 'query_one':
                    ret = count(image2image_val, save_id + '_image')
                    if ret == 1:
                        # query只生成一个
                        save_path = os.path.join(image_query_path, save_id)
                    else:
                        # 有则直接返回
                        return

                if not os.path.isdir(save_path):
                    os.mkdir(save_path)

                x1, y1, x2, y2 = this_data["box"]
                frame_after = frame[y1:y2, x1:x2, :]
                num_files = len(os.listdir(save_path))
                save_file = os.path.join(save_path, str(num_files) + '.jpg')
                add_sectence_embedding(save_path, folder_id, num_files, 'video')
                cv2.imwrite(save_file, frame_after)

    cap.release()
    cv2.destroyAllWindows()
    return 0


def generate_image2image_dataset(json_path, data_type, choices=''):
    # generate_image2image_dataset(os.path.join(image_annotation_file,j), v, image_annotation_path, 'train')
    # 根据商品图片生成图片类数据集
    with open(json_path) as f:
        data_ = json.load(f)
        # 打开json文件
        img_name = data_["img_name"]
        item_id = data_["item_id"]
        item_annotation = data_["annotations"][0]
        item_instance_id = str(item_annotation["instance_id"])
        img_files = os.path.join(image_path, item_id, img_name)
        img = cv2.imread(img_files)
        x1, y1, x2, y2 = item_annotation["box"]
        img_after = img[y1: y2, x1: x2, :]
        # choices=['gallery', 'query', 'train', 'val', 'gallery_or_query', 'train_or_val'])
        # 全部指定为某一类型的数据集
        save_id = item_id
        if data_type == 'gallery_all':
            save_path = os.path.join(image_gallery_path, save_id)

        if data_type == 'query_all':
            save_path = os.path.join(image_query_path, save_id)

        if data_type == 'train_all':
            save_path = os.path.join(image_train_path, save_id)

        if data_type == 'val_all':
            save_path = os.path.join(image_val_path, save_id)

        if data_type == 'gallery_first':
            ret = count(image2image_query, save_id+'_image')
            if ret <= image_train_or_gallery_num:
                # gallery分配不足，优先分配gallery
                save_path = os.path.join(image_gallery_path, save_id)
            else:
                # gallery分配充足，可以分配给query
                save_path = os.path.join(image_query_path, save_id)

        if data_type == 'query_first':
            ret = count(image2image_query, save_id +'_image')
            if ret <= image_val_or_query_num:
                save_path = os.path.join(image_query_path, save_id)
                # query分配不足，优先分配query
            else:
                # query分配充足，可以分配给gallery
                save_path = os.path.join(image_gallery_path, save_id)

        if data_type == 'train_first':
            ret = count(image2image_val, save_id +'_image')
            if ret <= image_train_or_gallery_num:
                # train分配不足，优先分配train
                save_path = os.path.join(image_train_path, save_id)
            else:
                # train分配充足，可以分配给val
                save_path = os.path.join(image_val_path, save_id)

        if data_type == 'val_first':
            ret = count(image2image_val, save_id+'_image')
            if ret <= image_val_or_query_num:
                # val分配不足，优先分配val
                save_path = os.path.join(image_val_path, save_id)
            else:
                # val分配充足，可以分配给train
                save_path = os.path.join(image_train_path, save_id)

        if data_type == 'query_one':
            ret = count(image2image_val, save_id+'_image')
            if ret == 1:
                # query只生成一个
                save_path = os.path.join(image_val_path, save_id)
            else:
                # 有则直接返回
                return
        # 如果不存在则创建
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        num_files = len(os.listdir(save_path))

        add_sectence_embedding(save_path, str(item_id), num_files, 'image')
        save_file = os.path.join(save_path, str(num_files) + '.jpg')
        cv2.imwrite(save_file, img_after)

    return 1




# <block/>全局变量定义

# 1.生成全语料库的词频表（商品库非视频库）


data_list = os.listdir(image_annotation_path)
frequency_dict = dict()  # 记录词频的字典
if 'diction.pkl' not in os.listdir('./') or 0:
    print('强制重新统计词频或此前不存在')
    for i, v in enumerate(data_list):
        image_text_file = os.path.join(image_text_path, v+'.txt')# 第一层：对每个商品进行遍历
        with open(image_text_file) as f_:
            index_list_str = f_.read()
            index_list = index_list_str.split(',')
            index_list = numpy.array(index_list)
            for item_index in index_list:
                if len(item_index) == 0:
                    vec_i = -1
                else:
                    vec_i = int(item_index)
                    if vec_i in frequency_dict:
                        frequency_dict[vec_i] += 1
                    else:
                        frequency_dict[vec_i] = 1


    with open('./diction.pkl', 'wb') as f:
        pickle.dump(frequency_dict, f)
else:
    frequency_file = open("./diction.pkl", "rb")  # 读取词频文件
    frequency_dict = pickle.load(frequency_file)
    frequency_file.close()
    print('从文件中加载词频统计')

dict_file = open("./all_index2vec_dict.pkl", "rb")  # 读取词向量字典
index2vec_dict = pickle.load(dict_file)
dict_file.close()
print('词向量字典加载完成')

# 2.生成train与val
print("开始处理训练的商品视频")
data_list = os.listdir(video_annotation_path)
# for id in tqdm(data_list[0:2]):
for id in tqdm(data_list[train_start:train_end]):
    data_annotation_file = os.path.join(video_annotation_path, id)
    check(data_annotation_file)
    with open(data_annotation_file) as f:
        data = json.load(f)
        for j in range(10):
            data_annotation = data["frames"][j]["annotations"]
            frame_index = data["frames"][j]["frame_index"]
            video_path_ = os.path.join(video_path, id[:-5] + ".mp4")
            generate_frame_image(video_path_, frame_index, data_annotation, 'val_first', folder_id=id[:-5],
                                 choices=['gallery_all', 'query_all', 'train_all', 'val_all', 'query_one'
                                          'gallery_first', 'query_first', 'train_first', 'val_first'])


print("开始处理训练数据的商品图片")
data_list = os.listdir(image_annotation_path)
# for i, v in tqdm(enumerate(data_list[0:2])):
for i, v in tqdm(enumerate(data_list[train_start:train_end])):
    image_annotation_file = os.path.join(image_annotation_path, v)  # 获取每件商品的每个图片的路径
    if not os.path.isdir(image_annotation_file):
        continue
    data_list = os.listdir(image_annotation_file)  # json文件的路径集合
    for j in data_list:
        data_annotation_file = os.path.join(image_annotation_file, j)
        check(data_annotation_file)
        generate_image2image_dataset(data_annotation_file, 'train_all',
                                     choices=['gallery_all', 'query_all', 'train_all', 'val_all', 'query_one'
                                              'gallery_first', 'query_first', 'train_first', 'val_first'])

# 3.生成gallery（商品库商品）与query（视频关键帧）

# 3.1 gallery图片
print("开始处理测试数据的商品图片")
data_list = os.listdir(image_annotation_path)
# for i, v in tqdm(enumerate(data_list[2:4])):
for i, v in tqdm(enumerate(data_list[test_start:test_end])):
    image_annotation_file = os.path.join(image_annotation_path, v)  # 获取每件商品的每个图片的路径
    if not os.path.isdir(image_annotation_file):
        continue
    data_list = os.listdir(image_annotation_file)  # json文件的路径集合
    for j in data_list:
        data_annotation_file = os.path.join(image_annotation_file, j)
        check(data_annotation_file)
        generate_image2image_dataset(data_annotation_file,
                                     'gallery_all',
                                     choices=['gallery_all', 'query_all', 'train_all', 'val_all', 'query_one'
                                              'gallery_first', 'query_first', 'train_first', 'val_first'])


# 3.2 query视频关键帧
print("开始处理测试数据的商品视频")
data_list = os.listdir(video_annotation_path)
# for id in tqdm(data_list[2:4]):
for id in tqdm(data_list[test_start:test_end]):
    data_annotation_file = os.path.join(video_annotation_path, id)
    check(data_annotation_file)
    with open(data_annotation_file) as f:
        data = json.load(f)
        for j in range(10):
            data_annotation = data["frames"][j]["annotations"]
            frame_index = data["frames"][j]["frame_index"]
            video_path_ = os.path.join(video_path, id[:-5] + ".mp4")
            generate_frame_image(video_path_, frame_index, data_annotation, 'query_one', folder_id=id[:-5],
                                 choices=['gallery_all', 'query_all', 'train_all', 'val_all', 'query_one'
                                          'gallery_first', 'query_first', 'train_first', 'val_first'])

print('所有数据预处理完成')
