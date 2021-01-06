import numpy as np

class DataInput:
  """
  整形済み学習・テストデータから指定されたバッチを生成
  Iterable & Iterator
  """
  def __init__(self, data, batch_size, imgs, img_list, texts):
    """
    data：学習またはテストデータ
    batch_size：バッチサイズ
    """
    self.batch_size = batch_size
    self.data = data
    self.imgs = imgs
    self.img_list = img_list
    self.texts = texts
    # エポック数の計算
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    # 現在のバッチインデックス
    self.i = 0

  def __iter__(self):
    """イテレーターとして自分自身を返す"""
    return self

  def __next__(self):
    """
    イテレーターが呼ばれた
    次のバッチを返す
    """
    if self.i == self.epoch_size:
      # データが尽きた
      raise StopIteration

    # このバッチに関係のあるデータをスライス
    # 「tsには行動の一覧」がリストになっている、一つの問題と同じ
    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    # 次のバッチインデックスへ
    self.i += 1

    # 行列へ入れ込む
    # u:ユーザーID
    # i:予測する商品のID(ラベル)
    # y:正例なら1、負例なら0
    # sl:履歴の長さ
    u, i, y, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[3])
      y.append(t[4])
      sl.append(len(t[1]))
    
    # 最大の履歴の長さに合わせて行列を初期化する
    max_sl = max(sl)
    
    # hist_i:行動履歴
    hist_i = np.zeros([len(ts), max_sl], np.int32)
    # hist_t:行動の時間（hist_i[i]の時間）
    # 行動時間は最後の行動からの相対時間になっている
    hist_t = np.zeros([len(ts), max_sl], np.int32)
    im = np.zeros((len(ts), max_sl, self.imgs.shape[-1]), dtype=np.float32)
    r = np.zeros((len(ts), max_sl, self.texts.shape[-1]), dtype=np.float32)

    # hist_iとhist_tに内容を書き込み
    k = 0
    for t in ts:
      # t[1]は行動の履歴
      # t[2]は行動の時間の履歴
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
        hist_t[k][l] = t[2][l]
        im[k][l] = self.imgs[self.img_list[t[1][l]]]
        r[k][l] = self.texts[t[1][l]]
      
      k += 1

    return self.i, (u, i, y, hist_i, hist_t, sl, im, r)

class DataInputTest:
  """DataInputのテストデータバージョン"""
  def __init__(self, data, batch_size, imgs, img_list, texts):

    # epoch_sizeを決定
    self.batch_size = batch_size
    self.data = data
    self.imgs = imgs
    self.img_list = img_list
    self.texts = texts
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  # 次の状態(i)とデータを出力
  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    # 今回入力するデータを現在の状態iに従って持ってくる
    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    # 次の状態
    self.i += 1

    # 今回入力するデータを整形
    # DataInputと違うのはこの部分
    # u:ユーザーID
    # i:正例のラベル(予測する商品のID)
    # j:負例のラベル(予測する商品のID)
    # sl:履歴の長さ
    u, i, j, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[3][0])
      j.append(t[3][1])
      sl.append(len(t[1]))
    max_sl = max(sl)
    
    #hist_i、hist_tの形を入力長で固定させる
    hist_i = np.zeros([len(ts), max_sl], np.int32)
    hist_t = np.zeros([len(ts), max_sl], np.int32)
    im = np.zeros((len(ts), max_sl, self.imgs.shape[-1]), dtype=np.float32)
    r = np.zeros((len(ts), max_sl, self.texts.shape[-1]), dtype=np.float32)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
        hist_t[k][l] = t[2][l]
        im[k][l] = self.imgs[self.img_list[t[1][l]]]
        r[k][l] = self.texts[t[1][l]]
      k += 1

    return self.i, (u, i, j, hist_i, hist_t, sl, im, r)