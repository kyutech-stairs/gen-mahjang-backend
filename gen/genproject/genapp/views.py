from django.shortcuts import render
from django.http import HttpResponse
import json
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Net

import numpy as np
from mahjong.shanten import Shanten
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.meld import Meld
import random
import chainer

@csrf_exempt
def index(request):
    # return HttpResponse("Hello, world. You're at the polls index.")

    if request.method == 'GET':
        return JsonResponse({})

    # JSON文字列
    datas = json.loads(request.body)

    #全37種
    syurui = {'1m':0,'2m':1,'3m':2,'4m':3,'5m':4,'6m':5,'7m':6,'8m':7,'9m':8,'1s':9,'2s':10,'3s':11,'4s':12,'5s':13,'6s':14,'7s':15,'8s':16,'9s':17,'1p':18,'2p':19,'3p':20,'4p':21,'5p':22,'6p':23,'7p':24,'8p':25,'9p':26,'a':27,'b':28,'c':29,'d':30,'e':31,'f':32,'g':33, 'a5m':34, 'a5s':35, 'a5p':36}
    #風4種
    field = {'a':0,'b':1,'c':2,'d':3}
    #萬子
    dic_man = {'1m':1, '2m':2, '3m':3, '4m':4, '5m':5, '6m':6, '7m':7, '8m':8, '9m':9, 'a5m':5}
    #索子
    dic_sou = {'1s':1, '2s':2, '3s':3, '4s':4, '5s':5, '6s':6, '7s':7, '8s':8, '9s':9, 'a5s':5}
    #筒子
    dic_pin = {'1p':1, '2p':2, '3p':3, '4p':4, '5p':5, '6p':6, '7p':7, '8p':8, '9p':9, 'a5p':5}
    #字牌
    dic_honors = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7}

    dora = datas["dora"]["name"]
    ground = datas["ground"]["name"]
    own = datas["own"]["name"]

    #配牌リスト
    pai_list = [0 for i in range(37)]
    for i in datas["haipai"]:
        pai_list[syurui[i["hai"]]] += i["amount"]

    input_list = pai_list

    #シャンテン数
    man = ''
    sou = ''
    pin = ''
    honors = ''
    for i in datas["haipai"]:
        if 'm' in i["hai"]:
            for j in range(i["amount"]):
                man += str(dic_man[i["hai"]])
        elif 's' in i["hai"]:
            for j in range(i["amount"]):
                sou += str(dic_sou[i["hai"]])
        elif 'p' in i["hai"]:
            for j in range(i["amount"]):
                pin += str(dic_pin[i["hai"]])
        else:
            for j in range(i["amount"]):
                honors += str(dic_honors[i["hai"]])
    shanten = Shanten()
    tiles = TilesConverter.string_to_34_array(man=man, pin=pin, sou=sou, honors=honors)
    ss = shanten.calculate_shanten(tiles)

    ##萬子,索子,筒子,字牌,中張牌,么九牌,ドラのカウント
    dd = 0
    m = 0
    s = 0
    p = 0
    h = 0
    tyu = 0
    yao = 0
    for i in datas["haipai"]:
        #ドラ枚数
        if i["hai"] == dora or i["hai"] == 'a5m' or i["hai"] == 'a5s' or i["hai"] == 'a5p':
            dd += i["amount"]
        #萬子枚数
        if 'm' in i["hai"]:
            m += i["amount"]
        #索子枚数
        elif 's' in i["hai"]:
            s += i["amount"]
        #筒子枚数
        elif 'p' in i["hai"]:
            p += i["amount"]
        #字牌枚数
        else:
            h += i["amount"]
        #么九牌
        if '1' in i["hai"] or '9' in i["hai"] or 'a' in i["hai"] or 'b' in i["hai"] or 'c' in i["hai"] or 'd' in i["hai"] or 'e' in i["hai"] or 'f' in i["hai"] or 'g' in i["hai"]:
            yao += i["amount"]
        else:
            tyu += i["amount"]

    input_list.append(dd)
    input_list.append(ss)
    input_list.append(m)
    input_list.append(s)
    input_list.append(p)
    input_list.append(h)
    input_list.append(tyu)
    input_list.append(yao)

    #ドラリスト化
    dora_list = [0 for i in range(34)]
    dora_list[syurui[dora]] += 1

    input_list[len(input_list):len(input_list)] = dora_list

    #場風リスト化
    field1_list = [0 for i in range(4)]
    field1_list[field[ground]] += 1

    input_list[len(input_list):len(input_list)] = field1_list

    #自風リスト化
    field2_list = [0 for i in range(4)]
    field2_list[field[own]] += 1

    input_list[len(input_list):len(input_list)] = field2_list

    input_data = []
    input_data.append(input_list)
    input_data = np.array(input_data)
    input_data = input_data.astype('float32')

    ##訓練済みネットワークを用いた推論
    #保存したネットワークの読み込み
    #訓練済みのネットワークと同様のクラスのインスタンス生成
    loaded_net = Net(n_hidden=200)

    #訓練済みネットワークのパラメータを読み込ませる
    chainer.serializers.load_npz('./genapp/ml_models/m_data.net', loaded_net)

    # テストデータで予測値を計算
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = loaded_net(input_data)

    resignation = np.argmax(y[0,:].array)

    point = random.randint(1,5)

    result = {
        'res':str(resignation),
        'point':str(point)
    }

    response = JsonResponse(
        result
    )
    return response
