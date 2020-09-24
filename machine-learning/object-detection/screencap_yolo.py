import cv2
import matplotlib.pyplot as plt
plt.ion()
from utils import *
from darknet import Darknet

from multiprocessing import Process, Queue

#class ClearableQueue(Queue):
#
#    def clear(self):
#        last = None
#        try:
#            while True:
#                last = self.get_nowait()
#        except:
#            pass
#        return last

import mss
import mss.tools
import time
import numpy as np

def pred(inqueue, outqueue):
    # Set the NMS Threshold
    score_threshold = 0.6
    # Set the IoU threshold
    iou_threshold = 0.4
    cfg_file = "cfg/yolov3.cfg"
    weight_file = "weights/yolov3.weights"
    namesfile = "data/coco.names"
    m = Darknet(cfg_file)
    m.load_weights(weight_file)
    class_names = load_class_names(namesfile)
    # m.print_network()
    t = time.time()
    tot_images = 0

    while "there are images":
      original_image = np.ones((1,1,3))
      try:
        while True: # Consume all but the last entry in the queue; drop frames if we're too slow.
          print("Prediction input queue is not empty, getting the next item...")
          original_image = inqueue.get_nowait()
      except:
        if original_image is not None and original_image.shape[0] == 1:
          time.sleep(0.05)
          continue
      #original_image = inqueue.clear()
      if original_image is None:
        print("Received poison pill; terminating prediction process...")
        plt.close('all')
        break
      print("Processing image of dimensions", original_image.shape)
      img = cv2.resize(original_image, (m.width, m.height))
      # detect the objects
      boxes = detect_objects(m, img, iou_threshold, score_threshold)
      # plot the image with the bounding boxes and corresponding object class labels
      plt.clf()
      plot_boxes(original_image, boxes, class_names, plot_labels=True)
      plt.pause(0.001)
      te = time.time() - t
      tot_images += 1
      print("Elapsed:", te, "Detection frames per second (fps):", tot_images/te)
      outqueue.put(original_image)

def grab(queue):
    t = time.time()

    # type: (Queue) -> None

    rect = {"top": 0, "left": 0, "width": 3840, "height": 2160}
    rect = {"top": 400, "left": 0, "width": 400, "height": 400}

    with mss.mss() as sct:
        for _ in range(1000):
            queue.put(cv2.cvtColor(np.array(sct.grab(rect)), cv2.COLOR_BGRA2BGR))
            te = time.time() - t
            print("Elapsed:", te, "frames per second (fps):", _/te)

    # Tell the other worker to stop
    queue.put(None)


def save(queue):
    # type: (Queue) -> None
    t = time.time()

    number = 0
    output = "screenshots/file_{}.png"
    to_png = mss.tools.to_png

    while "there are screenshots":
        img = queue.get()
        if img is None:
            break

        to_png(img.rgb, img.size, output=output.format(number))
        number += 1
        te = time.time() - t
        print("Elapsed:", te, "images saved per second:", number/te)


if __name__ == "__main__":
    # The screenshots queue
    predqueue = Queue()  # type: Queue
    savequeue = Queue()  # type: Queue

    # 2 processes: one for grabing and one for saving PNG files
    Process(target=grab, args=(predqueue,)).start()
    Process(target=pred, args=(predqueue,savequeue)).start()
    #Process(target=save, args=(savequeue,)).start()

