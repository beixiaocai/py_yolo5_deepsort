from yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker

import os
import time
import numpy as np
import warnings
import cv2
import torch
# from torch.backends import cudnn
import torch.backends.cudnn as cudnn

import sys

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

cudnn.benchmark = True


class TrackerParams(object):
    input_path = "test.mp4"
    save_path = None
    frame_interval = 0
    fourcc = None
    device = None
    save_txt = None

    display = False
    display_width = 720
    display_height = 480

    # YOLO-V5 parameters
    weights = None  # 'yolov5/weights/yolov5s.pt'
    img_size = 640
    conf_thres = 0.5
    iou_thres = 0.5
    classes = [0]
    agnostic_nms = False
    augment = False

    # deepsort parameters
    config_deepsort = None  # "./configs/deep_sort.yaml"

    # YOLO-V5 parameters
    # parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deepsort parameters
    # parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")


class VideoTracker(object):
    def __init__(self, params):
        print('VideoTracker.__init__')
        # ***************** Initialize ******************************************************
        params.img_size = check_img_size(params.img_size)

        self.params = params
        self.img_size = self.params.img_size  # image size in detector, default is 640
        self.frame_interval = self.params.frame_interval  # frequency
        self.device = select_device(self.params.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # create video capture ****************
        if self.params.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", self.params.display_width, self.params.display_height)

        # ***************************** initialize DeepSORT **********************************
        cfg = get_config()
        cfg.merge_from_file(self.params.config_deepsort)
        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        # ***************************** initialize YOLO-V5 **********************************
        self.detector = torch.load(self.params.weights, map_location=self.device)['model'].float()  # load to FP32
        self.detector.to(self.device).eval()
        if self.half:
            self.detector.half()  # to FP16

        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

        print('Done..')
        if self.device == 'cpu':
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.vdo = cv2.VideoCapture(-1)
        self.vdo.open(self.params.input_path)

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('Done. Load video file ', self.params.input_path)

        # ************************* create output *************************
        if self.params.save_path:
            os.makedirs(self.params.save_path, exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.params.save_path, "results.mp4")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.params.fourcc)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
            print('Done. Create output file ', self.save_video_path)

        if self.params.save_txt:
            os.makedirs(self.params.save_txt, exist_ok=True)

    def __del__(self):
        print('VideoTracker.__del__')
        self.vdo.release()
        self.writer.release()

    def run(self):

        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_outputs = None
        is_success = False

        while self.vdo.grab():
            # Inference *********************************************************************
            r, img0 = self.vdo.retrieve()
            if r:
                t0 = time.time()
                if idx_frame % self.params.frame_interval == 0:
                    __ret,outputs, yt, st = self.image_track(img0)  # (#ID, 5) x1,y1,x2,y2,id
                    if __ret:
                        last_outputs = outputs
                        is_success = True
                        yolo_time.append(yt)
                        sort_time.append(st)
                        print('Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
                else:
                    outputs = last_outputs  # directly use prediction in last frames

                if is_success:
                    t1 = time.time()
                    avg_fps.append(t1 - t0)

                    # post-processing ***************************************************************
                    # visualize bbox  ********************************
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        img0 = draw_boxes(img0, bbox_xyxy, identities)  # BGR

                        # add FPS information on output video
                        text_scale = max(1, img0.shape[1] // 1600)
                        cv2.putText(img0, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)),
                                    (20, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

                    # display on window ******************************
                    if self.params.display:
                        cv2.imshow("test", img0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            cv2.destroyAllWindows()
                            break

                    # save to video file *****************************
                    if self.params.save_path:
                        self.writer.write(img0)

                    if self.params.save_txt:
                        with open(self.params.save_txt + str(idx_frame).zfill(4) + '.txt', 'a') as f:
                            for i in range(len(outputs)):
                                x1, y1, x2, y2, idx = outputs[i]
                                f.write('{}\t{}\t{}\t{}\t{}\n'.format(x1, y1, x2, y2, idx))

                idx_frame += 1


        print('Avg YOLO time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),
                                                                      sum(sort_time) / len(sort_time)))
        t_end = time.time()
        print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

    def image_track(self, im0):
        """
        :param im0: original image, BGR format
        :return:
        """
        __ret = False
        # preprocess ************************************************************
        # Padded resize
        img = letterbox(im0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # numpy to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        s = '%gx%g ' % img.shape[2:]  # print string

        # Detection time *********************************************************
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.detector(img, augment=self.params.augment)[0]  # list: bz * [ (#obj, 6)]

        # Apply NMS and filter object other than person (cls:0)
        pred = non_max_suppression(pred, self.params.conf_thres, self.params.iou_thres,
                                   classes=self.params.classes, agnostic=self.params.agnostic_nms)
        t2 = time_synchronized()

        # get all obj ************************************************************
        det = pred[0]  # for video, bz is 1
        if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls

            # Rescale boxes from img_size to original im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results. statistics of number of each obj
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

            bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
            confs = det[:, 4:5].cpu()

            # ****************************** deepsort ****************************
            outputs = self.deepsort.update(bbox_xywh, confs, im0)
            # (#ID, 5) x1,y1,x2,y2,track_ID
            # print("deepsort:", outputs)
            if type(outputs) == list and len(outputs) == 0:
                pass
            else:
                __ret = True

        else:
            # outputs = torch.zeros((0, 5))
            outputs = []

        t3 = time.time()
        return __ret,outputs, t2 - t1, t3 - t2
