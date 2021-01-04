import os
import os.path as osp
import numpy as np
import pickle

top_k = 30
vis_root = 'http://yq01-sys-hic-k8s-v100-box-a223-0062.yq01.baidu.com:8903/cartoontoface/extraction/data/test'
data_root = osp.abspath(osp.dirname(osp.dirname(__file__)))

gallery_path='../data/gd/gallery.txt'
query_path='../data/gd/query.txt'
imgdir='../data/test'
data_dir='/home/shibaorong/cartoon/datasets/data/cartoon'
ranks = np.load("../train/multimodal/rankstrain.npy")

def get_query_and_gallery(data_path):

    C = []
    P = []

    for index, name in enumerate(sorted(os.listdir(data_path))):
        imgroot = os.path.join(data_path, name)
        for imgname in os.listdir(imgroot):
            imgpath = os.path.join(imgroot, imgname)
            if imgname[0] == 'C':
                C.append((index, imgpath))
            else:
                P.append((index, imgpath))

    return C,P

def visualize():
    C, P = get_query_and_gallery(data_path=data_dir)
    outhtml = osp.join(data_root, "searchanalysis.html")
    html_file = open(outhtml, "w")
    html_file.write('<html><meta charset=\"utf-8\"><body>\n')
    html_file.write('<p>\n')
    html_file.write('<table border="1">\n')
    querys = [i[1] for i in C]
    gallerys = [i[1] for i in P]

    for i in range(len(querys)):

        test_img = querys[i]
        html_file.write('<td><img src="%s" width="150" height="150" /></td>' % (test_img))
        html_file.write("<td> %s %s</td>\n" % (i, 'before'))
        html_file.write("<td> %s </td>\n" % (test_img))
        html_file.write("<td> %s </td>\n" % (gallerys[ranks[0, i]]))
        for j in range(top_k):
            index_img = gallerys[ranks[j, i]]
            html_file.write(
                    '<td><img style="border:4px solid yellow;", src="%s" width="150" height="150" /></td>' % (
                                index_img))

        html_file.write('</tr>\n')
        html_file.write('</tr>\n')

    html_file.write("</table>\n</p>\n</body>\n</html>")
    print("draw html finished ")


if __name__ == "__main__":

    visualize()