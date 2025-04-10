import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This automatically initializes CUDA driver
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Allocate buffers
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream

# Run inference
def do_inference(context, bindings, inputs, outputs, stream, input_image):
    # Copy input image to host
    np.copyto(inputs[0][0], input_image.ravel())

    # Transfer to GPU
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    # Execute model
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer prediction back
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    # Return prediction as numpy array
    return outputs[0][0]

# Preprocess the frame (resize and normalize)
def preprocess(frame, input_shape=(640, 640)):
    original_h, original_w = frame.shape[:2]
    
    # Calculate resize ratio while maintaining aspect ratio
    ratio = min(input_shape[0] / original_h, input_shape[1] / original_w)
    new_h, new_w = int(original_h * ratio), int(original_w * ratio)
    
    # Resize the image
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create a black canvas with target dimensions
    canvas = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)
    
    # Calculate padding
    dw, dh = (input_shape[1] - new_w) // 2, (input_shape[0] - new_h) // 2
    
    # Place the resized image on the canvas
    canvas[dh:dh+new_h, dw:dw+new_w, :] = resized
    
    # Store scaling info for later use
    preprocess_info = {
        'ratio': ratio,
        'dw': dw,
        'dh': dh,
        'new_h': new_h,
        'new_w': new_w,
        'original_shape': (original_h, original_w)
    }
    
    # Normalize and convert to proper format
    img = canvas.astype(np.float32) / 255.0  # normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # add batch dim
    
    return np.ascontiguousarray(img), preprocess_info


def xywh2xyxy(bboxes):
    """
        Convert nx4 boxes from [center x, center y, w, h, conf, class_id] to [x1, y1, x2, y2, conf, class_id]
    """
    out = bboxes.copy()
    out[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # top left x
    out[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # top left y
    out[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # bottom right x
    out[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # bottom right y
    return out


def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        indexes = np.where(iou <= threshold)[0]
        order = order[indexes + 1]
    keep = np.array(keep).astype(int)
    return keep


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_boxes_to_original(boxes, preprocess_info):
    """
    Scale boxes from model input size to original image size
    """
    ratio = preprocess_info['ratio']
    dw = preprocess_info['dw']
    dh = preprocess_info['dh']
    original_shape = preprocess_info['original_shape']
    
    # Remove padding
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    
    # Scale to original size
    boxes[:, :4] /= ratio
    
    # Clip to original image boundaries
    clip_coords(boxes, original_shape)
    
    return boxes


def postprocess(prediction, conf_thres, iou_thres, preprocess_info, input_shape=(640, 640)):
    xc = prediction[4:].max(0) > conf_thres
    prediction = prediction.transpose((1, 0))
    x = prediction[xc]

    if not x.shape[0]:
        return np.empty((0, 6), dtype=np.float32)

    box = x[:, :4]
    cls = x[:, 4:]

    i, j = np.where(cls > conf_thres)
    x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None]), 1)

    bboxes = xywh2xyxy(x)

    # ⚠️ Do NOT scale here — model already gives 0–1 coordinates relative to padded image
    # bboxes[:, :4] *= np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])

    labels = set(bboxes[:, 5].astype(int))

    detected_objects = []
    for label in labels:
        selected_bboxes = bboxes[np.where(bboxes[:, 5] == label)]
        selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4], iou_thres)]
        detected_objects += selected_bboxes_keep.tolist()

    if detected_objects:
        detected_objects = np.array(detected_objects)
        detected_objects[:, :4] = scale_boxes_to_original(detected_objects[:, :4], preprocess_info)
    else:
        return np.empty((0, 6), dtype=np.float32)

    return detected_objects


def draw_boxes(image, detections, class_names):
    for x1, y1, x2, y2, conf, class_id in detections:
        class_id = int(class_id)
        label = f"{class_names[class_id]}: {conf:.2f}"
        
        # Generate a color based on class ID
        color = (
            int(hash(str(class_id)) % 180 + 50),
            int(hash(str(class_id*2)) % 180 + 50),
            int(hash(str(class_id*3)) % 180 + 50)
        )

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Add filled background to text for better visibility
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, 
                     (int(x1), int(y1) - text_size[1] - 5), 
                     (int(x1) + text_size[0], int(y1)), 
                     color, -1)
        
        cv2.putText(image, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def log_gpu_memory(tag=""):
    free, total = cuda.mem_get_info()
    used = total - free
    print(f"[GPU MEM][{tag}] Used: {used / (1024**2):.2f} MB | Free: {free / (1024**2):.2f} MB | Total: {total / (1024**2):.2f} MB")

# --- Main Code ---
def main():
    engine_path = "yolo11s.engine"
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    input_shape = (640, 640)  # Standard YOLO input shape

    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video path
    if not cap.isOpened():
        print("Error: Cannot open camera or video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame and get preprocessing info for scaling boxes back later
        start_pre = time.perf_counter()
        input_tensor, preprocess_info = preprocess(frame, input_shape)
        end_pre = time.perf_counter()
        
        log_gpu_memory("Before Inference")

        # Run inference
        start_inf = time.perf_counter()
        pred = do_inference(context, bindings, inputs, outputs, stream, input_tensor)
        end_inf = time.perf_counter()

        log_gpu_memory("After Inference")
        
        # Postprocess
        start_post = time.perf_counter()
        pred = pred.reshape((84, 8400))
        
        ## Postprocess detections, properly scale back to original image
        detections = postprocess(pred, 0.25, 0.45, preprocess_info, input_shape)
        end_post = time.perf_counter()
        
        # Logging
        print(f"[PROFILE] Preprocess: {(end_pre - start_pre) * 1000:.2f} ms")
        print(f"[PROFILE] Inference: {(end_inf - start_inf) * 1000:.2f} ms")
        print(f"[PROFILE] Postprocess: {(end_post - start_post) * 1000:.2f} ms")
        
        # Draw the properly scaled boxes on the original frame
        frame = draw_boxes(frame, detections, COCO_CLASSES)
        
        # Display FPS
        fps_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv11s Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()