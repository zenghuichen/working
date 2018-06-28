"""
评价 cifar-100
cifar 差不多在100k之后达到0.83的机型读
speed
差不多，使用K40或着350-600images/sec，在100k步之后，精度：0.86
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import sys
import time
import os
import numpy as np
import tensorflow as tf
sys.path.append(r'E:\githubcode\cifar100')
#加载原有的模型结构
import model as cifar10

FLAGS=None #方便管理其中的结点
Eval_dir='/home/chen/文档/mygithub/mycode/cifar100/eval'
log_dir='/home/chen/文档/mygithub/mycode/cifar100/log'
Evaldata_dir=r'E:\exampletrain\tf\test128.tf'
def logout(x):
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MkDir(log_dir)
    log_time=time.gmtime()
    log_name='evallog'+str(log_time.tm_year)+str(log_time.tm_mon)+str(log_time.tm_mday)
    f=os.path.join(log_dir,log_name)
    f=open(f,'a')
    f.write('\n'+str(x))
    f.close()

def eval_once(saver,summary_writer,top_k_op,summary_op,images,logits,labels):
    '''
    Run Eval once
    :param saver:
    :param summary_writer:
    :param top_k_op:  ????????????????????
    :param summary_op:
    :return:
    '''
    with tf.Session() as sess:
        ckpt=tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:# restore the checkpoint
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('NO checkpoint file found')
            return
        coord=tf.train.Coordinator()  # start the queue runner
        try:
            t1=str(datetime.datetime.now())
            print(t1)
            threads=[]
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,coord=coord,daemon=True,start=True))
            num_iter=int(math.ceil(FLAGS.num_examples/64))
            true_count=0
            total_sample_count=num_iter*64
            step=0
            while step <num_iter and not coord.should_stop():
                predictions=sess.run([top_k_op])
                #单张图片检验
                labelt=sess.run(labels)
                true_count+=np.sum(predictions) # 统计正确的结果
                step+=1

            # 计算精度
            precision=true_count/total_sample_count  # 计算正确率
            t='%s precision @1 =%f'%(datetime.datetime.now(),precision)
            print(t)  # 输出结果
            summary=tf.Summary() # 获取汇总结点值
            summary.ParseFromString(sess.run(summary_op))# ???????????????????
            summary.value.add(tag='Precision @1',simple_value=precision)
            t2=str(datetime.datetime.now())
            print(t2)
            summary_writer.add_summary(summary,global_step)
            logout(t1)
            logout(t)
            logout(t2)

        except Exception as e:
            coord.request_stop(e)
            print(str(e))

        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=10)#????????

def evaluate():
    with tf.Graph().as_default() as g:
        
        images,labels=cifar10.model_image_input(Evaldata_dir,True)
        logits=cifar10.inference(images)
        labels=tf.cast(labels,tf.int32)
        top_k_op =tf.nn.in_top_k(logits,labels,1)  #统计分类正确的结果类
        var_avg=tf.train.ExponentialMovingAverage(cifar10.moving_avg_decay)
        var_to_restore=var_avg.variables_to_restore()
        saver=tf.train.Saver(var_to_restore)

        summary_op=tf.summary.merge_all()
        summary_write=tf.summary.FileWriter(FLAGS.eval_dir,g) #输出 summary结果
        while True:
            eval_once(saver,summary_write,top_k_op,summary_op,images,logits,labels)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def evalmain(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--eval_dir',
      type=str,
      default=Eval_dir,
      help='Directory where to write event logs.')
  parser.add_argument(
      '--eval_data',
      type=str,
      default='test',
      help="""Either 'test' or 'train_eval'.""")
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default=cifar10.Checkoutpointdir,
      help="""Directory where to read model checkpoints.""")
  parser.add_argument(
      '--eval_interval_secs',
      type=int,
      default=60 * 5,
      help='How often to run the eval.')
  parser.add_argument(
      '--num_examples',
      type=int,
      default=750,
      help='Number of examples to run.')
  parser.add_argument(
      '--run_once',
      type=bool,
      default=False,
      help='Whether to run eval only once.')

  FLAGS, unparsed = parser.parse_known_args()
def maineval():
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

