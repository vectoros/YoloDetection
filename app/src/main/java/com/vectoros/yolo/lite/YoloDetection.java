package com.vectoros.yolo.lite;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;

import org.tensorflow.env.Logger;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Vector;

public abstract class YoloDetection implements Classifier {

    private static final Logger LOGGER = new Logger();

    // Only return this many results.
    private static final int NUM_DETECTIONS = 10;
    // Float model
    private static final float IMAGE_MAX = 255.0f;

    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;

    private static final int NUM_BOXES_PER_BLOCK = 3;

    private static final float NMS_THRESH = 0.4f;

    private boolean isModelQuantized;
    // Pre-allocated buffers.
    private Vector<String> labels;
    private int[] intValues = new int[getInputSize() * getInputSize()];

    private ByteBuffer imgData;

    private MappedByteBuffer tfliteModel;

    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    private Delegate gpuDelegate = null;

    private Interpreter tflite;

    private Map<Integer, Object> outputMap;

    // init in subclass
    int[] outputWidths;
    int[] anchors;
    int[][] mask;

    YoloDetection(Activity activity, boolean isModelQuantized) throws IOException {
        this.isModelQuantized = isModelQuantized;
        this.tfliteModel = loadModelFile(activity);
        this.tflite = new Interpreter(tfliteModel, tfliteOptions);
        this.imgData =
                ByteBuffer.allocateDirect(
                        BATCH_SIZE
                                * getInputSize()
                                * getInputSize()
                                * PIXEL_SIZE
                                * getNumBytesPerChannel(isModelQuantized));
        this.imgData.order(ByteOrder.nativeOrder());

        this.labels = loadLabelList(activity);
    }


    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(Activity activity)
            throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private Vector<String> loadLabelList(Activity activity) throws IOException {
        Vector<String> labelList = new Vector<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < getInputSize(); ++i) {
            for (int j = 0; j < getInputSize(); ++j) {
                final int val = intValues[pixel++];
                if (isModelQuantized) {
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) (val & 0xFF));
                } else {
                    imgData.putFloat(((val >> 16) & 0xFF) / IMAGE_MAX);
                    imgData.putFloat(((val >> 8) & 0xFF) / IMAGE_MAX);
                    imgData.putFloat((val & 0xFF) / IMAGE_MAX);
                }

            }
        }
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        long start = SystemClock.currentThreadTimeMillis();
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        convertBitmapToByteBuffer(bitmap);

        Trace.endSection(); // preprocessBitmap

        Trace.beginSection("runInference");
        runInference();

        Trace.endSection();
        final List<Recognition> recognitions = postProcess();

        Trace.endSection();
        long end = SystemClock.currentThreadTimeMillis();
        LOGGER.d("Time cost:%dms", end - start);
        return recognitions;
    }

    /**
     * Enables use of the GPU for inference, if available.
     */
    public void useGpu() {
        if (gpuDelegate == null && GpuDelegateHelper.isGpuDelegateAvailable()) {
            gpuDelegate = GpuDelegateHelper.createGpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
            LOGGER.d("Using GPU delegate");
        }
    }

    /**
     * Enables use of the CPU for inference.
     */
    public void useCPU() {
        tfliteOptions.setUseNNAPI(false);
        recreateInterpreter();
    }

    /**
     * Enables use of NNAPI for inference, if available.
     */
    public void useNNAPI(boolean flag) {
        LOGGER.d("Using NNAPI but not support, just use CPU");
        tfliteOptions.setUseNNAPI(flag);
        recreateInterpreter();
    }

    /**
     * Adjusts the number of threads used in CPU inference.
     */
    public void setNumThreads(int numThreads) {
        tfliteOptions.setNumThreads(numThreads);
        recreateInterpreter();
    }

    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            // TODO(b/120679982)
            // gpuDelegate.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    /**
     * Closes the interpreter and model to release resources.
     */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        tfliteModel = null;
    }

    private static float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    private static float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    private static float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    private static float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    private static void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    private static float sigmoid(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private List<Recognition> nms(List<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            NUM_DETECTIONS,
                            (lhs, rhs) -> {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClassId() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);

                //clear pq to do next nms
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < NMS_THRESH) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    private void runInference(){
        Object[] inputArray = {imgData};
        outputMap = new HashMap<>(getNumOutputTensor());
        for(int i = 0; i < getNumOutputTensor(); i++){
            // width * height * channel * boxesPerBlock * numClasses.
            float [][][][] out = new float[1][outputWidths[i]][outputWidths[i]][NUM_BOXES_PER_BLOCK * (5 + labels.size())];
            outputMap.put(i, out);
        }
        tflite.runForMultipleInputsOutputs(inputArray, outputMap);
    }

    private List<Recognition> postProcess(){
        final List<Recognition> recognitions = new ArrayList<>();
        for(int i = 0; i <getNumOutputTensor(); i++ ){
            int outputWidth = outputWidths[i];
            //output outputWidth * outputHeight * (boxesPerBlock * (labels.size + 5))
            float [][][] out = ((float [][][][]) Objects.requireNonNull(outputMap.get(i)))[0];

            final int labelSizeWithResult = labels.size() + 5;
            final float inputOutputAspectRatio = getInputSize() / (float)outputWidth;

            for(int y = 0; y < outputWidth; ++y){
                for(int x = 0; x < outputWidth; ++x)
                    for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {

                        final float confidence = sigmoid(out[y][x][b * labelSizeWithResult + 4]);
                        int detectedClass = -1;
                        float maxClass = 0;

                        final float[] classes = new float[labels.size()];
                        for (int c = 0; c < labels.size(); ++c) {
                            classes[c] = out[y][x][b * labelSizeWithResult + 5 + c];
                        }
                        softmax(classes);

                        for (int c = 0; c < labels.size(); ++c) {
                            if (classes[c] > maxClass) {
                                detectedClass = c;
                                maxClass = classes[c];
                            }
                        }

                        final float confidenceInClass = maxClass * confidence;

                        if (confidenceInClass > getObjectThresh()) {
                            final int baseIndex = b * labelSizeWithResult;
                            final float centerX = (x + sigmoid(out[y][x][baseIndex])) * inputOutputAspectRatio;
                            final float centerY = (y + sigmoid(out[y][x][baseIndex + 1])) * inputOutputAspectRatio;

                            final float w = (float) (Math.exp(out[y][x][baseIndex + 2]) * anchors[2 * mask[i][b]]);
                            final float h = (float) (Math.exp(out[y][x][baseIndex + 3]) * anchors[2 * mask[i][b] + 1]);

                            final RectF detection = new RectF(
                                    Math.max(0, centerX - w / 2),
                                    Math.max(0, centerY - h / 2),
                                    Math.min(getInputSize() - 1, centerX + w / 2),
                                    Math.min(getInputSize() - 1, centerY + h / 2));

                            recognitions.add(new Recognition("" + detectedClass, labels.get(detectedClass), confidenceInClass, detection, detectedClass));
                        }
                    }
            }
        }
        return nms(recognitions);
    }

    public abstract float getObjectThresh();

    public abstract int getInputSize();

    protected abstract int getNumOutputTensor();

    protected abstract int getNumBytesPerChannel(boolean isQuantized);

    protected abstract String getModelPath();

    protected abstract String getLabelPath();
}
