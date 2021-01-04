import os, sys
sys.path.insert(0,'/root/paddlejob/workspace/env_run/shibaorong/deglr')
import os.path as osp
import numpy as np
import pickle

top_k = 30
vis_root = 'http://yq01-sys-hic-k8s-v100-box-a223-0062.yq01.baidu.com:8903/cartoontoface/extraction/data/test'
data_root = osp.abspath(osp.dirname(osp.dirname(__file__)))

gallery_path='../data/cartoon/gallery.txt'
query_path='../data/cartoon/query.txt'
imgdir='/home/shibaorong/cartoon/extraction/data/cartoontest'

ranks = np.load("ranks.npy")

def get_img_name(path,imgdir):
    imgpaths=[]
    for line in open(path):
        imgpaths.append(os.path.join(imgdir,line.strip()+'.jpg'))
    return imgpaths

def visualize():
    outhtml = osp.join(data_root, "searchanalysis.html")
    html_file = open(outhtml, "w")
    html_file.write('<html><meta charset=\"utf-8\"><body>\n')
    html_file.write('<p>\n')
    html_file.write('<table border="1">\n')
    querys = get_img_name(query_path, imgdir)
    gallerys = get_img_name(gallery_path, imgdir)

    for i in range(len(querys)):

        test_img = querys[i]
        html_file.write('<td><img src="%s" width="150" height="150" /></td>' % (test_img))
        html_file.write("<td> %s %s</td>\n" % (i, 'before'))
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