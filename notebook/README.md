# やること

+ とりあえず実データに当てはめようとしたら、どの分布をどういう意図で使っているのか（信念）が異様に大事だということはわかった。（単にポワソンだとなんで？となりそう）
+ なので、Udemyみつつ、一色 numpyro の使い方確認したらその辺を注意して実装する
  + どのように AB　比較描画するのが良いか
  + ┗ これは Udemy で説明されてた
```
def model(y = None, g = None, num+data = 0):
    mu = numpyro.sample('mu', dist.HalfNormal(10), sample_shape = (2,)) # 二つデータ出す
    mu_dup = mu[g]
    with numpyro.plate('data', num_data):
        numpyro.sample('obs', dist.Poisson(mu_dup), obs = y)　# 2種類
    mu_diff = numpyro.deterministic('mu_diff', mu[1] = mu[0])
```
  + 指数分布の方が良さそうな場合あるか（これはWebで始める機械学習読んでて、滞在時間連続値のところで出てきたが、実際のDVCVの値はこれに連動するかもと思った）
  + 