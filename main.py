from VideoTracker import TrackerParams, VideoTracker

if __name__ == '__main__':

    params = TrackerParams
    params.input_path = "data/test.mp4"
    params.save_path = "data/output/"
    params.save_txt = "data/output/predict/"
    params.frame_interval = 2
    params.fourcc = "mp4v"
    params.device = "0"  # 'cuda device, i.e. 0 or 0,1,2,3 or cpu'

    params.display = True
    params.display_width = 720
    params.display_height = 480

    params.weights = "yolov5/weights/yolov5s.pt"
    params.img_size = 640
    params.conf_thres = 0.5
    params.iou_thres = 0.5
    params.classes = [0]
    params.agnostic_nms = False
    params.augment = False

    params.config_deepsort = "configs/deep_sort.yaml"

    tracker = VideoTracker(params)
    tracker.run()

