import requests
from lxml import etree
import re
from utils import *
import pandas as pd
from tkinter import _flatten  #好用的_flatten函数，可以方便地处理嵌套列表
Base_info_list = []
Distance_to_Metro_list = []
House_price_list = []
columns = ['rooms', 'area', 'num_floor', 'decoration_level', 'distance_to_metro', 'unit_price', '', 'house_price']
# dataframe = pd.DataFrame({'rooms': [], 'area': [], 'num_floor': [],
#                           'decoration_level': [], 'distance_to_metro': [],
#                           'unit_price': [], 'house_price': [], })
dataframe = pd.DataFrame(columns=columns)
CSV_INDEX=True
csv_path = r'G:\Python\PATHON_Practice\Machine_learning\Regression\My_regression'
headers = {
    'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
}
#
proxies = {
  "http": "http://10.10.1.10:3128",
  "https": "http://10.10.1.10:1080",
}
# response = requests.get(url,headers = headers)
s = requests.session()

# 在网页中查找元素
for num in range(1,301):
    url = f'https://bj.5i5j.com/ershoufang/n{num}/'
    html = requests.get(url,headers=headers) # 爬取到源代码
    html.encoding = 'utf-8'
    content = etree.HTML(html.text)
    #注意：不要用xpath的绝对路径，这样网站稍作改动就无效了
    Base_info_list = content.xpath("//div[@class='listX']/p[1]/text()")
    Distance_to_Metro_list = content.xpath("//div[@class='listX']/p[2]/text()")
    Unit_price_list = content.xpath("//div[@class='jia']/p[2]/text()")
    House_price_list = content.xpath("//div[@class='jia']/p[1]/strong/text()")


    Info_string = [Base_info.split('·') for Base_info in Base_info_list]
    Info_distance = [Distance.split('·')[-1] for Distance in Distance_to_Metro_list]
    Info_unit_price = [Unit_price.split(',') for Unit_price in Unit_price_list]
    Info_House_price = [price.split(',') for price in House_price_list]


    rooms = [int(re.findall(r"\d+\.?\d*", Info[0])[0])+int(re.findall(r"\d+\.?\d*", Info[0])[1])
             for Info in Info_string]
    # \d+ 匹配一个或多个小数点前的数字；  \.?匹配一个或零个小数点； \d*匹配[零个]或多个小数点后的数字
    area = [re.findall(r"\d+\.?\d*", Info[1]) for Info in Info_string]
    num_floor = [re.findall(r"\d+", Info[3]) for Info in Info_string]
    # 注意：有些商家没有标注装修情况——给出了一个很好的使用prprocesing中的imuter的例子，同时需要处理text attributes!
    decoration_level = [get_incomplete_data(Info,'decoration') for Info in Info_string]
    distance_to_metro = [get_incomplete_data(Info,'distance') for Info in Info_distance]
    unit_price = [re.findall(r"\d+", Info[0]) for Info in Info_unit_price]
    house_price = Info_House_price
    serie = pd.Series({
        'rooms':_flatten(rooms),
        'area':_flatten(area),
        'num_floor':_flatten(num_floor),
        'decoration_level':decoration_level,
        'distance_to_metro':_flatten(distance_to_metro),
        'unit_price':_flatten(unit_price),
        'house_price':_flatten(house_price)
    })
    # dataframe = dataframe.append(serie,ignore_index=True)##注意：要给DataFrame添加行，必须注明参数：igrore_index = True
    dataframe = dataframe.append(pd.DataFrame({  #如果用series的，得到的csv形状很奇怪
        'rooms':_flatten(rooms),
        'area':_flatten(area),
        'num_floor':_flatten(num_floor),
        'decoration_level':decoration_level,
        'distance_to_metro':_flatten(distance_to_metro),
        'unit_price':_flatten(unit_price),
        'house_price':_flatten(house_price)
    }),ignore_index=True)
    # if num is 1:
    #     CSV_INDEX = True
    # else:
    #     CSV_INDEX = False
dataframe.to_csv(csv_path+f'\house_price_chiyuan.csv',index=CSV_INDEX)









