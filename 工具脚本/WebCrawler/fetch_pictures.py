# -*- coding: utf-8 -*-

import urllib
import requests
import pdb
import re
import sys
import os
from os.path import join
def get_url_one_page(url):
    html = requests.get(url)
    html.encoding = 'utf-8'
    html = html.text
    url_pic_this_page = re.findall(r'"objURL":"(.*?)",', html)
    url_next_page_prefix = re.findall(r'<a href="(.*?)" class="n">下一页', html)
    if len(url_next_page_prefix)!=0:
        url_next_page = 'http://image.baidu.com' + url_next_page_prefix[0]
    else:
        print("已到达最后一页！")
        url_next_page=None
    return url_pic_this_page, url_next_page
def download_pics(url_pics, count_total,key):
    count_success = 0
    for url_pic in url_pics:
        try:
            pic = requests.get(url_pic, timeout=10)
            ext = url_pic.split('.')[-1]
            pic_name = join(SAVE_DIR, key+'_{}.'.format(count_success+1)+ext)
            with open(pic_name, 'wb') as f:
                f.write(pic.content)
            print('已下载图片: {}张'.format(count_success+1))
            count_success += 1
        except:
            print('第{}张图片下载失败！已跳过...'.format(count_success+1))
            continue
        if count_success+1>count_total:
            print('所有{}张图片下载完毕!'.format(count_success))
            return
def fetch_pictures(key, num_pics):
    print('[+]开始爬虫: 关键词：{}, 爬取图片数量：{} '.format(key, num_pics))
    url_init_base = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='
    url_init = url_init_base + urllib.parse.quote(key)
    #pdb.set_trace()
    url_pic_list = []
    while True:
        url_pic_this_page, url_next_page = get_url_one_page(url=url_init)
        url_pic_list += url_pic_this_page
        #url_pic_list.append([x for x in url_pic_this_page])
        if url_next_page is not None:
            url_init = url_next_page
        else:
            print('[+]图片页数已达最后！')
            break
        if len(url_pic_list)>num_pics+1:  #为了防止下载失败，多爬取两页的内容
    
            print('[+]爬虫结束！')
            break
    print('[+]开始下载图片, 请稍等...')
    download_pics(url_pic_list, num_pics,key)
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python fetch_picturs.py [key],[num_pics]")
    key = sys.argv[1]
    num_pics = int(sys.argv[2])
    SAVE_DIR = key
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    fetch_pictures(key, num_pics)