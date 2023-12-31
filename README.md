# AB_test

A/Bテストの実践に向け、関連して学んだ知見やコードスニペットを置いとくリポジトリ

## 目的

Bayesian A/B Testing について学ぶ

> 従来の頻度論ABテストから脱却するため取り入れた知識をメモしておく。<br>
> p-value や結果の解釈が一人歩きして辛くなるため、ベイズ推論を使った解釈の導入を進める。

## 主な参考図書
+ [ウェブ最適化ではじめる機械学習](https://www.amazon.co.jp/dp/4873119162)
+ [Sample Colab Code](https://colab.research.google.com/drive/19KX0LE8aVf4cQ7DDsvl6VoV_Psh9mQaM?usp=sharing#forceEdit=true&sandboxMode=true&scrollTo=v5_-ryxucFc9)

## その他参考サイトなど
+ [ベイズ推定を用いたA/Bテストの評価を実際に試してみた](https://www.ai-shift.co.jp/techblog/2952)
+ [ベイジアンA/Bテストの利点と実験計画策定に関する一検討](https://hack.nikkei.com/blog/advent20221216/)
+ [ベイジアンA/Bテストの使い方](https://exploratory.io/note/GMq1Qom5tS/A-B-wHL0xqZ0tm)
+ [ソウゾウにベイジアンA/Bテストを導入しました](https://engineering.mercari.com/blog/entry/20221110-bayesian-testing-for-souzoh/)
+ [ベイズ推論によるA/Bテストの効果検証入門](https://yosukeyoshida.netlify.app/posts/bayesian-ab-testing/)
+ [ベイジアンABテストのためにARPUのモデリングに挑戦してみた](https://inside.dmm.com/articles/bayesian-ab-testing-arpu/)

## なぜ頻度論よりベイジアンA/Bテストが優れているか？

| タイプ | メリット | デメリット |
| ---- | :--- | :--- |
| 頻度論 | ・実装が容易<br>・考え方として一定の市民権を得ており、p-value 0.05 を見れば納得できるなど解釈に慣れてる人が多い<br>| ・「有意な差はない（誤差の範囲である）」と判定されると意思決定できなくなる<br>・必要なサンプルサイズに届くまでに時間がかかる（”停止規則”に依存する）<br>・p-value 解釈の誤解が多い |
| ベイジアン |・単純な勝ち負け（無風）という結果ではなく、どの程度の割合でテストパターンが良いのかが分かる（解釈が容易）<br>・サンプルサイズを明確に計測せず途中経過を確認できる | ・考え方に慣れていない人が多い（ベイズモデリングの知識がないメンバーへの展開）<br>・新たなKPIを設定するたびに、事前分布を探す必要がある |
