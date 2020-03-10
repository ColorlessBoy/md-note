这里是我看的一些资料罗列（时间顺序），都是PyTorch官方资料中关于分布式并行训练的内容。我认为其中3、5、6这三个例程参考价值比较大。

1. PyTorch官方教程-并行与分布式训练1：Model Parallel Best Practices

   https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

   **主题**：网络模型的并行——手动将一个模型分割到多个GPU中训练。

2. PyTorch官方教程-并行与分布式训练2：Getting Started with Distributed Data Parallel

   https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

   **主题**：

   - `DistributedDataParallel(DDP)`模块的介绍与使用（多CPU多GPU）；
   - `DDP`模型的保存与加载；
   - DDP与模型并行结合使用。

3. PyTorch官方教程-并行与分布式训练3: Writing Distributed Applications with PyTorch

   https://pytorch.org/tutorials/intermediate/dist_tuto.html

   **主题**：

   - PyTorch中的并行化编程所需要的通讯API，即`torch.distributed`模块，可以直接传输`torch.Tensor`；
   - 一个分布式SGD的例程；
   - 进阶内容：PyTorch通讯API基于三个框架而来，分别是`Gloo`,`MPI`以及`NCCL`。传输CPU上的数据`Gloo`最好；传输GPU上的数据`NCCL`最好；如果想使用基于`MPI`的通讯API，需要源码编译PyTorch，并且性能是未知的。

4. 具体内容同时可以参考官方文档，PyTorch的两个模块：

   - `torch.distributed`: https://pytorch.org/docs/stable/distributed.html
   - `torch.multiprocessing`:https://pytorch.org/docs/stable/multiprocessing.html?highlight=multiprocess#module-torch.multiprocessing

5. 官方示例代码1-imagenet

   https://github.com/pytorch/examples/tree/master/imagenet

   **主题**：多节点多GPU下训练imagenet网络。

6. 官方示例代码2-mnist_hogwild

   https://github.com/pytorch/examples/tree/master/mnist_hogwild

   **主题**：单GPU多进程下训练CNN网络，torch.multiprocessing的教程。

