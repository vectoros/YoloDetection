package com.vectoros.yolo.lite;

import android.app.Activity;

import org.tensorflow.env.Logger;

import java.io.IOException;


public class Yolov3TinyDetection extends YoloDetection {
    private static final Logger LOGGER = new Logger();
    private static final int INPUT_SIZE = 320;
    private static final int MIN_OUTPUT_WIDTH = INPUT_SIZE / 32;
    private static final int NUM_OUTPUT_TENSORS = 2;

    public Yolov3TinyDetection(Activity activity, boolean isModelQuantized) throws IOException {
        super(activity, isModelQuantized);
        outputWidths = new int[]{MIN_OUTPUT_WIDTH, MIN_OUTPUT_WIDTH * 2, MIN_OUTPUT_WIDTH * 4};
        anchors = new int[]{
                10,14,  23,27,  37,58, 81,82,  135,169,  344,319
        };
        mask = new int[][]{{3,4,5},{0,1,2}};
    }

    @Override
    public float getObjectThresh() {
        return 0.1f;
    }

    @Override
    public int getInputSize() {
        return INPUT_SIZE;
    }

    @Override
    protected int getNumBytesPerChannel(boolean isQuantized) {
        return 4;
    }

    @Override
    protected int getNumOutputTensor() {
        return NUM_OUTPUT_TENSORS;
    }

    @Override
    protected String getModelPath() {
        return "yolov3_tiny_320.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "labelmap.txt";
    }

}
