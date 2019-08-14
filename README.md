
 
## 目录：

* [PyTorch学习教程、手册](#PyTorch学习教程手册)
* [PyTorch视频教程](#PyTorch视频教程)
* [NLP&PyTorch实战](#NLPPyTorch实战)
* [CV&PyTorch实战](#CVPyTorch实战)
* [PyTorch论文推荐](#PyTorch论文推荐)
* [Pytorch书籍推荐](#PyTorch书籍推荐)

## PyTorch学习教程、手册
 
* [PyTorch英文版官方手册](https://pytorch.org/tutorials/)：对于英文比较好的同学，非常推荐该PyTorch官方文档，一步步带你从入门到精通。该文档详细的介绍了从基础知识到如何使用PyTorch构建深层神经网络，以及PyTorch语法和一些高质量的案例。
* [PyTorch中文官方文档](https://github.com/fendouai/PyTorchDocs)：阅读上述英文文档比较困难的同学也不要紧，我们为大家准备了比较官方的PyTorch中文文档，文档非常详细的介绍了各个函数，可作为一份PyTorch的速查宝典。
* [比较偏算法实战的PyTorch代码教程](https://github.com/yunjey/pytorch-tutorial)：在github上有很高的star。建议大家在阅读本文档之前，先学习上述两个PyTorch基础教程。
* [开源书籍](https://github.com/zergtant/pytorch-handbook)：这是一本开源的书籍，目标是帮助那些希望和使用PyTorch进行深度学习开发和研究的朋友快速入门。但本文档不是内容不是很全，还在持续更新中。
* [简单易上手的PyTorch中文文档](https://github.com/fendouai/pytorch1.0-cn)：非常适合新手学习。该文档从介绍什么是PyTorch开始，到神经网络、PyTorch的安装，再到图像分类器、数据并行处理，非常详细的介绍了PyTorch的知识体系，适合新手的学习入门。该文档的官网：[http://pytorchchina.com](http://pytorchchina.com) 。

## PyTorch视频教程
* [B站PyTorch视频教程](https://www.bilibili.com/video/av31914351/)：首推的是B站中近期点击率非常高的一个PyTorch视频教程，虽然视频内容只有八集，但讲的深入浅出，十分精彩。只是没有中文字幕，小伙伴们是该练习一下英文了...
* [国外视频教程](https://www.youtube.com/watch?v=SKq-pmkekTk)：另外一个国外大佬的视频教程，在YouTube上有很高的点击率，也是纯英文的视频，有没有觉得外国的教学视频不管是多么复杂的问题都能讲的很形象很简单？
* [莫烦](https://morvanzhou.github.io/tutorials/machine-learning/torch/)：相信莫烦老师大家应该很熟了，他的Python、深度学习的系列视频在B站和YouTube上均有很高的点击率，该PyTorch视频教程也是去年刚出不久，推荐给新手朋友。
* [101学院](https://www.bilibili.com/video/av49008640/)：人工智能101学院的PyTorch系列视频课程，讲的比较详细、覆盖的知识点也比较广，感兴趣的朋友可以试听一下。
* [七月在线](https://www.julyedu.com/course/getDetail/140/)：最后，向大家推荐的是国内领先的人工智能教育平台——七月在线的PyTorch入门与实战系列课。课程虽然是收费课程，但课程包含PyTorch语法、深度学习基础、词向量基础、NLP和CV的项目应用、实战等，理论和实战相结合，确实比其它课程讲的更详细，推荐给大家。
 

## NLP&PyTorch实战
* [Pytorch text](https://github.com/pytorch/text)：Torchtext是一个非常好用的库，可以帮助我们很好的解决文本的预处理问题。此github存储库包含两部分：
    * torchText.data：文本的通用数据加载器、抽象和迭代器（包括词汇和词向量）
    * torchText.datasets：通用NLP数据集的预训练加载程序
我们只需要通过pip install torchtext安装好torchtext后，便可以开始体验Torchtext 的种种便捷之处。
* [Pytorch-Seq2seq](https://github.com/IBM/pytorch-seq2seq)：Seq2seq是一个快速发展的领域，新技术和新框架经常在此发布。这个库是在PyTorch中实现的Seq2seq模型的框架，该框架为Seq2seq模型的训练和预测等都提供了模块化和可扩展的组件，此github项目是一个基础版本，目标是促进这些技术和应用程序的开发。
* [BERT NER](https://github.com/kamalkraj/BERT-NER)：BERT是2018年google 提出来的预训练语言模型，自其诞生后打破了一系列的NLP任务，所以其在nlp的领域一直具有很重要的影响力。该github库是BERT的PyTorch版本，内置了很多强大的预训练模型，使用时非常方便、易上手。
* [Fairseq](https://github.com/pytorch/fairseq)：Fairseq是一个序列建模工具包，允许研究人员和开发人员为翻译、总结、语言建模和其他文本生成任务训练自定义模型，它还提供了各种Seq2seq模型的参考实现。该github存储库包含有关入门、训练新模型、使用新模型和任务扩展Fairseq的说明，对该模型感兴趣的小伙伴可以点击上方链接学习。
* [Quick-nlp](https://github.com/outcastofmusic/quick-nlp)：Quick-nlp是一个深受fast.ai库启发的深入学习Nlp库。它遵循与Fastai相同的API，并对其进行了扩展，允许快速、轻松地运行NLP模型。
* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)：这是OpenNMT的一个PyTorch实现，一个开放源码的神经网络机器翻译系统。它的设计是为了便于研究，尝试新的想法，以及在翻译，总结，图像到文本，形态学等许多领域中尝试新的想法。一些公司已经证明该代码可以用于实际的工业项目中，更多关于这个github的详细信息请参阅以上链接。
 
## CV&PyTorch实战
* [pytorch vision](https://github.com/pytorch/vision)：Torchvision是独立于pytorch的关于图像操作的一些方便工具库。主要包括：vision.datasets 、vision.models、vision.transforms、vision.utils 几个包，安装和使用都非常简单，感兴趣的小伙伴们可以参考以上链接。
* [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch)：此github库是OpenFace在Pytorch中的实现，代码要求输入的图像要与原始OpenFace相同的方式对齐和裁剪。
* [TorchCV](https://github.com/donnyyou/torchcv)：TorchCV是一个基于PyTorch的计算机视觉深度学习框架，支持大部分视觉任务训练和部署，此github库为大多数基于深度学习的CV问题提供源代码，对CV方向感兴趣的小伙伴还在等什么？
* [Pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune)：该github库是利用pytorch对预训练卷积神经网络进行微调，支持的架构和模型包括：ResNet 、DenseNet、Inception v3 、VGG、SqueezeNet 、AlexNet 等。
* [Pt-styletransfer](https://github.com/tymokvo/pt-styletransfer#pt-styletransfer)：这个github项目是Pytorch中的神经风格转换，具体有以下几个需要注意的地方：
    * StyleTransferNet作为可由其他脚本导入的类；
    * 支持VGG（这是在PyTorch中提供预训练的VGG模型之前）
    * 可保存用于显示的中间样式和内容目标的功能
    * 可作为图像检查图矩阵的函数
    * 自动样式、内容和产品图像保存
    * 一段时间内损失的Matplotlib图和超参数记录，以跟踪有利的结果
* [Face-alignment](https://github.com/1adrianb/face-alignment#face-recognition)：Face-alignment是一个用 pytorch 实现的 2D 和 3D 人脸对齐库，使用世界上最准确的面对齐网络从 Python 检测面部地标，能够在2D和3D坐标中检测点。该github库详细的介绍了使用Face-alignment进行人脸对齐的基本流程，欢迎感兴趣的同学学习。
 

## PyTorch论文推荐
* [Google_evolution](https://github.com/neuralix/google_evolution)：该论文实现了实现了由Esteban Real等人提出的图像分类器大规模演化的结果网络。在实验之前，需要我们安装好PyTorch、 Scikit-learn以及下载好 [CIFAR10 dataset数据集](https://www.cs.toronto.edu/~kriz/cifar.html)。
* [PyTorch-value-iteration-networks](https://github.com/onlytailei/Value-Iteration-Networks-PyTorch)：该论文基于作者最初的Theano实现和Abhishek Kumar的Tensoflow实现，包含了在PyTorch中实现价值迭代网络（VIN）。Vin在NIPS 2016年获得最佳论文奖。
* [Pytorch Highway](https://github.com/kefirski/pytorch_Highway)：Highway Netowrks是允许信息高速无阻碍的通过各层，它是从Long Short Term Memory(LSTM) recurrent networks中的gate机制受到启发，可以让信息无阻碍的通过许多层，达到训练深层神经网络的效果，使深层神经网络不在仅仅具有浅层神经网络的效果。该论文是Highway network基于Pytorch的实现。
* [Pyscatwave](https://github.com/edouardoyallon/pyscatwave)：Cupy/Pythorn的散射实现。散射网络是一种卷积网络，它的滤波器被预先定义为子波，不需要学习，可以用于图像分类等视觉任务。散射变换可以显著降低输入的空间分辨率（例如224x224->14x14），且双关功率损失明显为负。
* [Pytorch_NEG_loss](https://github.com/kefirski/pytorch_NEG_loss)：该论文是Negative Sampling Loss的Pytorch实现。Negative Sampling是一种求解word2vec模型的方法，它摒弃了霍夫曼树，采用了Negative Sampling（负采样）的方法来求解，本论文是对Negative Sampling的loss函数的研究，感兴趣的小伙伴可点击上方论文链接学习。
* [Pytorch_TDNN](https://github.com/kefirski/pytorch_TDNN)：该论文是对Time Delayed NN的Pytorch实现。论文详细的讲述了TDNN的原理以及实现过程。
 
## PyTorch书籍推荐
相较于目前Tensorflow类型的书籍已经烂大街的状况，PyTorch类的书籍目前已出版的并没有那么多，笔者给大家推荐我认为还不错的四本PyTorch书籍。
* **《深度学习入门之PyTorch》**，电子工业出版社，作者：廖星宇。这本《深度学习入门之PyTorch》是所有PyTorch书籍中出版的相对较早的一本，作者以自己的小白入门深度学习之路，深入浅出的讲解了PyTorch的语法、原理以及实战等内容，适合新手的入门学习。但不足的是，书中有很多不严谨以及生搬硬套的地方，需要读者好好甄别。
推荐指数：★★★
* **《PyTorch深度学习》**，人民邮电出版社，作者：王海玲、刘江峰。该书是一本英译书籍，原作者是两位印度的大佬，该书除了PyTorch基本语法、函数外，还涵盖了ResNET、Inception、DenseNet等在内的高级神经网络架构以及它们的应用案例。该书适合数据分析师、数据科学家等相对有一些理论基础和实战经验的读者学习，不太建议作为新手的入门选择。
推荐指数：★★★
* **《深度学习框架PyTorch入门与实践》**，电子工业出版社，作者：陈云。这是一本2018年上市的PyTorch书籍，包含理论入门和实战项目两大部分，相较于其它同类型书籍，该书案例非常的翔实，包括：Kaggle竞赛中经典项目、GAN生成动漫头像、AI滤镜、RNN写诗、图像描述任务等。理论+实战的内容设置也更适合深度学习入门者和从业者学习。
推荐指数：★★★★
* **《PyTorch机器学习从入门到实战》**，机械工业出版社，作者：校宝在线、孙琳等。该书同样是一本理论结合实战的Pytorch教程，相较于前一本入门+实战教程，本书的特色在于关于深度学习的理论部分讲的非常详细，后边的实战项目更加的综合。总体而言，本书也是一本适合新手学习的不错的PyTorch入门书籍。
推荐指数：★★★



