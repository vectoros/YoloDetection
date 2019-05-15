# YoloDetection
Running Yolov3 and Yolov3-tiny on Andorid

# How to use

1. Download yolov3 darknet cfg and weights from [Darknet](https://pjreddie.com/darknet/yolo/)
2. Convert `cfg` and `weights` to keras `.h5` model
3. Convert `.h5` model to `.tflite` model
4. Put `.tflite` files to app/src/assets/
5. Modify model `filename` in `Yolov3Detection.java` and `Yolov3TinyDetection.java`

# Test

1. Only test on Pixel with Android Pie
