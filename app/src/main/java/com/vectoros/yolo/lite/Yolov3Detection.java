package com.vectoros.yolo.lite;

import android.app.Activity;

import org.tensorflow.env.Logger;

import java.io.IOException;


public class Yolov3Detection extends YoloDetection {
    private static final Logger LOGGER = new Logger();
    private static final int INPUT_SIZE = 320;
    private static final int MIN_OUTPUT_SIZE = INPUT_SIZE / 32;
    private static final int NUM_OUTPUT_TENSORS = 3;

    public Yolov3Detection(Activity activity, boolean isModelQuantized) throws IOException {
        super(activity, isModelQuantized);
        outputWidths = new int[]{MIN_OUTPUT_SIZE, MIN_OUTPUT_SIZE * 2, MIN_OUTPUT_SIZE * 4};
        anchors = new int[]{
                10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
        };
        mask = new int[][]{{6,7,8},{3,4,5},{0,1,2}};
    }

    @Override
    public float getObjectThresh() {
        return 0.3f;
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
        return "yolov3_320.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "labelmap.txt";
    }

}
