import os
import urllib.request


def get_file(root_path, all_files={}):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件，获取其path
    '''
    files = os.listdir(root_path)
    for idx,file in enumerate(files):
        if idx == 5:
            break
        if not os.path.isdir(root_path + '/' + file):   # not a dir
            all_files[file] = root_path + '/' + file
        else:  # is a dir
            get_file((root_path+'/'+file), all_files)
    return all_files




def get_pic_by_url(folder_path, lists):
    if not os.path.exists(folder_path):
        print("Selected folder not exist, try to create it.")
        os.makedirs(folder_path)
    for idx,url in enumerate(lists):
        if idx == 5:
            break
        print("Try downloading file: {}".format(url))
        filename = url.split('/')[-1]
        filepath = folder_path + '/' + filename
        if os.path.exists(filepath):
            print("File have already exist. skip")
        else:
            try:
                urllib.request.urlretrieve(url, filename=filepath)
            except Exception as e:
                print("Error occurred when downloading file, error message:")
                print(e)


if __name__ == "__main__":
    root_path = '/data1/yangzhenbang_new/datasets/SBU'
    paths = get_file(root_path)
    print(paths)
    for filename, path in paths.items():
        print('reading file: {}'.format(filename))
        with open(path, 'r') as f:
            lines = f.readlines()
            url_list = []
            for line in lines:
                url_list.append(line.strip('\n'))
            foldername = "./picture_get_by_url/pic_download/{}".format(filename.split('.')[0])
            get_pic_by_url(foldername, url_list)



# if __name__ == '__main__':
#     path = "/data1/yangzhenbang_new/datasets/SBU"
#     print(get_file(path))

